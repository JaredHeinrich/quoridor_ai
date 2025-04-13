use crate::error::LoggerError;
use crate::models::LogEntry;
use crate::traits::GenerationLike;
use anyhow::Result;
use lazy_static::lazy_static;
use neural_network::neural_network::NeuralNetwork;
use serde_json::{from_reader, to_writer};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::sync::Mutex;
use std::time::UNIX_EPOCH;

// Global cache of file indexes with last modification time
lazy_static! {
    static ref LINE_INDEX_CACHE: Mutex<HashMap<String, (Vec<u64>, u128)>> =
        Mutex::new(HashMap::new());
}

// Helper function to handle logging logic
fn write_to_log<T: serde::Serialize + ?Sized>(data: &T, output_path: &str) -> Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)
        .map_err(|e| LoggerError::FileOpenError(output_path.to_string(), e.to_string()))?;

    to_writer(&mut file, data).map_err(|e| LoggerError::SerializationError(e.to_string()))?;

    // Add a newline for better readability
    writeln!(file).map_err(|e| LoggerError::WriteError(output_path.to_string(), e.to_string()))?;

    Ok(())
}

// Logs a single neural network to a file in JSON format.
pub fn log_single_log_entry(log_entry: &LogEntry, output_path: &str) -> Result<()> {
    write_to_log(log_entry, output_path)
}

// Logs a generation of neural networks to a file in JSON format.
pub fn log_several_log_entries(log_entries: &[LogEntry], output_path: &str) -> Result<()> {
    write_to_log(log_entries, output_path)
}

// Reads and deserializes log entries from a file.
pub fn read_log_entries(output_path: &str) -> Result<Vec<LogEntry>> {
    let file = File::open(output_path)
        .map_err(|e| LoggerError::FileOpenError(output_path.to_string(), e.to_string()))?;
    let reader = BufReader::new(file);

    let log_entries: Vec<LogEntry> =
        from_reader(reader).map_err(|e| LoggerError::DeserializationError(e.to_string()))?;

    Ok(log_entries)
}

// logs a single neural network with placement and generation_index
pub fn log_single_network(
    generation_index: usize,
    placement: usize,
    neural_network: NeuralNetwork,
    fitness: Option<f64>,
    output_path: &str,
) -> Result<()> {
    let log_entry = LogEntry {
        generation_index,
        placement,
        neural_network,
        fitness,
    };
    log_single_log_entry(&log_entry, output_path)
}

/// Logs an entire Generation with placements based on fitness scores
/// (highest fitness gets first placement)
pub fn log_generation(generation: &mut impl GenerationLike, output_path: &str) -> Result<()> {
    // Sort the generation by fitness
    generation.sort_by_fitness()?;

    let log_entries: Vec<LogEntry> = generation
        .agents()
        .iter()
        .enumerate()
        .map(|(index, agent)| LogEntry {
            generation_index: generation.generation_index(),
            placement: index,
            neural_network: agent.neural_network().clone(),
            fitness: agent.fitness(),
        })
        .collect();

    log_several_log_entries(&log_entries, output_path)
}

/// Read specific lines from a file using indexed access
pub fn read_specific_lines(line_numbers: &[usize], output_path: &str) -> Result<Vec<LogEntry>> {
    // Sort and deduplicate line numbers
    let mut sorted_lines = line_numbers.to_vec();
    sorted_lines.sort_unstable();
    sorted_lines.dedup();

    // Get line positions, using cache if possible
    let line_positions = get_cached_line_index(output_path)?;
    let mut result = Vec::with_capacity(sorted_lines.len());
    let file = File::open(output_path)?;
    let mut reader = BufReader::new(file);

    // Read each requested line
    for &line_num in &sorted_lines {
        if line_num >= line_positions.len() {
            return Err(anyhow::anyhow!("Line number out of bounds").into());
        }

        // Seek to line position
        reader.seek(SeekFrom::Start(line_positions[line_num]))?;

        // Read the line
        let mut line = String::new();
        reader.read_line(&mut line)?;

        // Parse the JSON
        if !line.trim().is_empty() {
            let entry: LogEntry = serde_json::from_str(line.trim())?;
            result.push(entry);
        }
    }

    Ok(result)
}

// Get cached line index or build a new one
fn get_cached_line_index(file_path: &str) -> Result<Vec<u64>> {
    let metadata = std::fs::metadata(file_path)?;
    let modified = metadata.modified()?.duration_since(UNIX_EPOCH)?.as_millis();

    // Try to get from cache first
    let mut cache = LINE_INDEX_CACHE.lock().unwrap();
    if let Some((positions, last_modified)) = cache.get(file_path) {
        if *last_modified == modified {
            return Ok(positions.clone());
        }
    }

    // Not in cache or modified, build new index
    let positions = build_line_index(file_path)?;
    cache.insert(file_path.to_string(), (positions.clone(), modified));

    Ok(positions)
}

/// Build an index of byte positions for each line
fn build_line_index(file_path: &str) -> Result<Vec<u64>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut positions = vec![0]; // First line starts at position 0
    let mut pos = 0;

    for line in reader.lines() {
        let line_len = line?.len() as u64 + 1; // +1 for newline
        pos += line_len;
        positions.push(pos);
    }

    Ok(positions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::LogEntry;
    use std::fs;
    use std::io::Read;
    use tempfile::NamedTempFile;

    #[test]
    fn test_log_single_network() {
        let neural_network =
            NeuralNetwork::new(&vec![3, 2, 1]).expect("Failed to create neural network");
        let log_entry = LogEntry {
            generation_index: 1,
            placement: 4,
            neural_network,
            fitness: Some(0.95),
        };
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let output_path = temp_file.path().to_str().unwrap();

        log_single_log_entry(&log_entry, output_path).expect("Failed to log single network");

        // Read the file and verify the content
        let mut file = File::open(output_path).expect("Failed to open temp file");
        let mut content = String::new();
        file.read_to_string(&mut content)
            .expect("Failed to read temp file");

        let logged_entry: LogEntry =
            serde_json::from_str(&content.trim()).expect("Failed to deserialize log entry");
        assert_eq!(logged_entry.generation_index, log_entry.generation_index);
        assert_eq!(logged_entry.placement, log_entry.placement);
        assert_eq!(
            logged_entry.neural_network.layer_sizes,
            log_entry.neural_network.layer_sizes
        );
        assert_eq!(logged_entry.fitness, log_entry.fitness);
    }

    #[test]
    fn test_log_generation() {
        let log_entries = vec![
            LogEntry::new(
                1,
                10,
                NeuralNetwork::new(&vec![4, 4, 4]).expect("Failed to create neural network"),
            ),
            LogEntry::new(
                2,
                20,
                NeuralNetwork::new(&vec![2, 2]).expect("Failed to create neural network"),
            ),
        ];
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let output_path = temp_file.path().to_str().unwrap();

        log_several_log_entries(&log_entries, output_path).expect("Failed to log generation");

        // Read the file and verify the content
        let mut file = File::open(output_path).expect("Failed to open temp file");
        let mut content = String::new();
        file.read_to_string(&mut content)
            .expect("Failed to read temp file");

        let logged_entries: Vec<LogEntry> =
            serde_json::from_str(&content.trim()).expect("Failed to deserialize log entries");
        assert_eq!(logged_entries, log_entries);
    }

    #[test]
    fn test_read_log_entries() {
        let log_entries = vec![
            LogEntry::new(
                1,
                10,
                NeuralNetwork::new(&vec![4, 4, 4]).expect("Failed to create neural network"),
            ),
            LogEntry::new(
                2,
                20,
                NeuralNetwork::new(&vec![2, 2]).expect("Failed to create neural network"),
            ),
        ];
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let output_path = temp_file.path().to_str().unwrap();

        // Write log entries to the file
        let mut file = File::create(output_path).expect("Failed to create temp file");
        serde_json::to_writer(&mut file, &log_entries).expect("Failed to write log entries");

        // Read the log entries back
        let read_entries = read_log_entries(output_path).expect("Failed to read log entries");
        assert_eq!(read_entries, log_entries);
    }

    #[test]
    fn test_read_log_entries_invalid_file() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let output_path = temp_file.path().to_str().unwrap();

        // Write invalid data to the file
        fs::write(output_path, "invalid json").expect("Failed to write invalid data");

        // Attempt to read the log entries and expect an error
        let result = read_log_entries(output_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_specific_lines() {
        // Create a temporary file with multiple log entries
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let output_path = temp_file.path().to_str().unwrap();

        // Create several log entries
        let log_entries = vec![
            LogEntry::new(
                0,
                0,
                NeuralNetwork::new(&vec![2, 2]).expect("Failed to create neural network"),
            ),
            LogEntry::new(
                1,
                1,
                NeuralNetwork::new(&vec![2, 2]).expect("Failed to create neural network"),
            ),
            LogEntry::new(
                2,
                2,
                NeuralNetwork::new(&vec![2, 2]).expect("Failed to create neural network"),
            ),
        ];

        // Write each entry on its own line
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .open(output_path)
            .expect("Failed to open file");

        for entry in &log_entries {
            serde_json::to_writer(&mut file, entry).expect("Failed to write entry");
            writeln!(file).expect("Failed to write newline");
        }

        // Test reading specific lines
        let read_entries =
            read_specific_lines(&[0, 2], output_path).expect("Failed to read specific lines");
        assert_eq!(read_entries.len(), 2);
        assert_eq!(read_entries[0].generation_index, 0);
        assert_eq!(read_entries[1].generation_index, 2);

        // Test reading with duplicates (should deduplicate)
        let read_entries =
            read_specific_lines(&[1, 1, 1], output_path).expect("Failed to read specific lines");
        assert_eq!(read_entries.len(), 1);
        assert_eq!(read_entries[0].generation_index, 1);

        // Test reading out-of-bounds line
        let result = read_specific_lines(&[5], output_path);
        assert!(result.is_err());

        // Test reading empty selection
        let read_entries =
            read_specific_lines(&[], output_path).expect("Failed to read specific lines");
        assert!(read_entries.is_empty());
    }

    #[test]
    fn test_build_line_index() {
        // Test with multiple lines
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let output_path = temp_file.path().to_str().unwrap();

        fs::write(output_path, "line1\nline2\nline3").expect("Failed to write test data");

        let positions = build_line_index(output_path).expect("Failed to build line index");
        assert_eq!(positions.len(), 4); // 3 lines + end position
        assert_eq!(positions[0], 0); // First line starts at position 0
        assert_eq!(positions[1], 6); // After "line1\n"
        assert_eq!(positions[2], 12); // After "line2\n"

        // Test with empty file
        let empty_file = NamedTempFile::new().expect("Failed to create temp file");
        let empty_path = empty_file.path().to_str().unwrap();

        let positions =
            build_line_index(empty_path).expect("Failed to build line index for empty file");
        assert_eq!(positions.len(), 1); // Just the starting position

        // Test with non-existent file
        let result = build_line_index("non_existent_file.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_cached_line_index() {
        // Create a temporary file
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let output_path = temp_file.path().to_str().unwrap();

        fs::write(output_path, "line1\nline2\nline3").expect("Failed to write test data");

        // First call should build the index
        let first_call = get_cached_line_index(output_path).expect("Failed to get line index");
        assert_eq!(first_call.len(), 4);

        // Second call should use cached index
        let second_call =
            get_cached_line_index(output_path).expect("Failed to get cached line index");
        assert_eq!(first_call, second_call); // Should be identical (from cache)

        // Modify the file to invalidate cache
        std::thread::sleep(std::time::Duration::from_millis(100)); // Ensure modified time differs
        fs::write(output_path, "line1.0\nline2\nline3\nline4").expect("Failed to update test data");
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Should rebuild the index because file was modified
        let third_call =
            get_cached_line_index(output_path).expect("Failed to get updated line index");
        assert_eq!(third_call.len(), 5); // Now 4 lines + end position
        assert_ne!(first_call, third_call);
    }

    #[test]
    fn test_read_specific_lines_empty_file() {
        // Create an empty temporary file
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let output_path = temp_file.path().to_str().unwrap();

        // Try to read from empty file
        let result =
            read_specific_lines(&[0], output_path).expect("Should succeed with empty file");
        assert!(result.is_empty());

        // Reading no lines should succeed
        let read_entries =
            read_specific_lines(&[], output_path).expect("Failed to read empty selection");
        assert!(read_entries.is_empty());
    }
}

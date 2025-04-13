use crate::error::LoggerError;
use crate::models::LogEntry;
use crate::traits::GenerationLike;
use anyhow::Result;
use neural_network::neural_network::NeuralNetwork;
use serde_json::{from_reader, to_writer};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Write, Seek, SeekFrom, BufRead};
use std::sync::{Mutex};
use std::collections::HashMap;
use std::time::UNIX_EPOCH;
use lazy_static::lazy_static;

// Global cache of file indexes with last modification time
lazy_static! {
    static ref LINE_INDEX_CACHE: Mutex<HashMap<String, (Vec<u64>, u64)>> = Mutex::new(HashMap::new());
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
pub fn log_generation(
    generation: &mut impl GenerationLike,
    output_path: &str,
) -> Result<()> {
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
pub fn read_specific_lines(
    line_numbers: &[usize],
    output_path: &str
) -> Result<Vec<LogEntry>> {
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
    let modified = metadata.modified()?.duration_since(UNIX_EPOCH)?.as_secs();
    
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
}

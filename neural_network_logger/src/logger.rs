use crate::error::LoggerError;
use crate::models::LogEntry;
use anyhow::Result;
use neural_network::neural_network::NeuralNetwork;
use serde_json::{from_reader, to_writer};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Write};
use evolution_training::evolution::generation::Generation;

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
    output_path: &str,
) -> Result<()> {
    let log_entry = LogEntry {
        generation_index,
        placement,
        neural_network,
    };
    log_single_log_entry(&log_entry, output_path)
}

// Logs a generation of neural networks. It assmumes that the neural networks are in order with the first neural network being the best.
pub fn log_generation(
    generation_index: usize,
    ordered_networks: Vec<NeuralNetwork>,
    output_path: &str,
) -> Result<()> {
    let log_entries: Vec<LogEntry> = ordered_networks
        .into_iter()
        .enumerate()
        .map(|(index, network)| LogEntry {
            generation_index,
            placement: index as usize,
            neural_network: network,
        })
        .collect();

    log_several_log_entries(&log_entries, output_path)
}

/// Logs an entire Generation with placements based on fitness scores
/// (highest fitness gets first placement)
pub fn log_generation_from_struct(
    generation_index: usize,
    generation: &Generation,
    output_path: &str,
) -> Result<()> {
    // Create a sorted copy of the generation
    let mut sorted_generation = generation.clone();
    sorted_generation.sort_by_fitness()?;
    
    // Extract neural networks in order of fitness
    let networks: Vec<NeuralNetwork> = sorted_generation
        .agents
        .iter()
        .map(|agent| agent.neural_network.clone())
        .collect();
    
    // Use existing function to log the generation
    log_generation(generation_index, networks, output_path)
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

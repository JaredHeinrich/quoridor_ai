use serde_json::to_writer;
use std::fs::OpenOptions;
use std::io::{self, Write};
use crate::models::LogEntry;
use crate::error::LoggerError;

// Helper function to handle logging logic
fn write_to_log<T: serde::Serialize>(data: &T, output_path: &str) -> Result<(), LoggerError> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)
        .map_err(|e| LoggerError::FileOpenError(output_path.to_string(), e.to_string()))?;

    to_writer(&mut file, data)
        .map_err(|e| LoggerError::SerializationError(e.to_string()))?;

    // Add a newline for better readability
    writeln!(file)
        .map_err(|e| LoggerError::WriteError(output_path.to_string(), e.to_string()))?;

    Ok(())
}

// Logs a single neural network to a file in JSON format.
pub fn log_single_network(log_entry: &LogEntry, output_path: &str) -> Result<(), LoggerError> {
    write_to_log(log_entry, output_path)
}

// Logs a generation of neural networks to a file in JSON format.
pub fn log_generation(log_entries: &[LogEntry], output_path: &str) -> Result<(), LoggerError> {
    write_to_log(log_entries, output_path)
}
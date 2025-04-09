use thiserror::Error;

#[derive(Error, Debug)]
pub enum LoggerError {
    #[error("Error opening file '{0}': {1}")]
    FileOpenError(String, String),

    #[error("Error serializing data to JSON: {0}")]
    SerializationError(String),

    #[error("Error writing newline to file '{0}': {1}")]
    WriteError(String, String),
}
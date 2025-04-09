// This file defines the public interface for the logging functionality of the neural network logger crate.

pub mod logger;
pub mod models;
pub mod error;

pub use logger::{log_single_network, log_generation, read_log_entries};
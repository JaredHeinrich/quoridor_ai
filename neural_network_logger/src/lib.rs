// This file defines the public interface for the logging functionality of the neural network logger crate.

pub mod error;
pub mod logger;
pub mod models;

pub use logger::{log_generation, log_single_network, read_log_entries};
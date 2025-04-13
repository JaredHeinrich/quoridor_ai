use neural_network::neural_network::NeuralNetwork;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct LogEntry {
    pub generation_index: usize,
    pub placement: usize,
    pub neural_network: NeuralNetwork,
}

impl LogEntry {
    pub fn new(generation_index: usize, placement: usize, neural_network: NeuralNetwork) -> Self {
        LogEntry {
            generation_index,
            placement,
            neural_network,
        }
    }
}

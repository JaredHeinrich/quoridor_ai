use serde::{Serialize, Deserialize};
use neural_network::neural_network::NeuralNetwork;

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct LogEntry {
    pub generation: usize,
    pub placement: usize,
    pub neural_network: NeuralNetwork,
}

impl LogEntry {
    pub fn new(generation: usize, placement: usize, neural_network: NeuralNetwork) -> Self {
        LogEntry {
            generation,
            placement,
            neural_network,
        }
    }
}
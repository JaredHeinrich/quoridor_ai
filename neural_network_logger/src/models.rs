use serde::Serialize;
use neural_network::neural_network::NeuralNetwork;
use matrix::matrix::Matrix;

#[derive(Serialize)]
pub struct LogEntry {
    pub generation: usize,
    pub performance_ranking: f64,
    pub neural_network: NeuralNetwork,
}
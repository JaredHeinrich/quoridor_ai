use anyhow::Result;
use neural_network::neural_network::NeuralNetwork;

/// Trait for agent-like objects that have a neural network and fitness
pub trait AgentLike {
    /// Get a reference to the agent's neural network
    fn neural_network(&self) -> &NeuralNetwork;

    /// Get the agent's fitness score, if available
    fn fitness(&self) -> Option<f64>;
}

/// Trait for generation-like objects that contain agents
pub trait GenerationLike {
    /// Get the generation's index/number
    fn generation_index(&self) -> usize;

    /// Get a slice of all agents in the generation
    fn agents(&self) -> Vec<&dyn AgentLike>;

    /// Sort the generation's agents by fitness (highest first)
    fn sort_by_fitness(&mut self) -> Result<()>;
}

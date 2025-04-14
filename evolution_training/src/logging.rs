use neural_network_logger::traits::{AgentLike, GenerationLike};
use crate::evolution::agent::Agent;
use crate::evolution::generation::Generation;
use anyhow::Result;

// Implement the AgentLike trait for Agent
impl AgentLike for Agent {
    fn neural_network(&self) -> &neural_network::neural_network::NeuralNetwork {
        &self.neural_network
    }
    
    fn fitness(&self) -> Option<f64> {
        Some(self.fitness)
    }
}

// Implement the GenerationLike trait for Generation
impl GenerationLike for Generation {
    fn generation_index(&self) -> usize {
        self.generation_index
    }
    
    fn agents(&self) -> Vec<&dyn AgentLike> {
        self.agents.iter()
            .map(|agent| agent as &dyn AgentLike)
            .collect()
    }
    
    fn sort_by_fitness(&mut self) -> Result<()> {
        // Implement sorting directly to avoid name conflict
        self.agents.sort_by(|a, b| {
            b.fitness.partial_cmp(&a.fitness).unwrap()
        });
        
        Ok(())
    }
}
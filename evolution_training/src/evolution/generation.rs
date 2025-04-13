use anyhow::Result;
use neural_network::neural_network::NeuralNetwork;

use crate::settings::Settings;
use crate::evolution::agent::Agent;

/// Represents a generation of neural network agents for evolutionary training
#[derive(Debug, Clone)]
pub struct Generation {
        pub agents: Vec<Agent>,
}

impl Generation {
    /// Creates a new initial generation of neural networks based on settings
    pub fn create_initial(settings: &Settings) -> Result<Self> {
        settings.validate()?;
        
        let mut agents = Vec::with_capacity(settings.generation_size);
        
        for _ in 0..settings.generation_size {
            let neural_network = NeuralNetwork::new(&settings.neural_network_layer_structure)?;
            agents.push(Agent { neural_network, fitness: 0.0 });
        }
        
        Ok(Generation { agents })
    }
    
    /// Returns the number of agents in this generation
    pub fn size(&self) -> usize {
        self.agents.len()
    }
    
    /// Gets a reference to a specific neural network by index
    pub fn get_neural_network(&self, index: usize) -> Option<&NeuralNetwork> {
        self.agents.get(index).map(|agent| &agent.neural_network)
    }
    
    /// Gets a mutable reference to a specific neural network by index
    pub fn get_neural_network_mut(&mut self, index: usize) -> Option<&mut NeuralNetwork> {
        self.agents.get_mut(index).map(|agent| &mut agent.neural_network)
    }
    
    /// Sets fitness scores for the agents in this generation
    pub fn set_fitness_scores(&mut self, scores: Vec<f64>) -> Result<()> {
        if scores.len() != self.agents.len() {
            return Err(anyhow::anyhow!("Number of fitness scores doesn't match number of agents"));
        }
        
        for (agent, score) in self.agents.iter_mut().zip(scores) {
            agent.fitness = score;
        }
        
        Ok(())
    }
    
    /// Sort agents by fitness score (highest first)
    pub fn sort_by_fitness(&mut self) -> Result<()> {        
        self.agents.sort_by(|a, b| {
            b.fitness.partial_cmp(&a.fitness).unwrap()
        });
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_initial_generation() {
        let settings = Settings::default();
        let generation = Generation::create_initial(&settings);
        
        assert!(generation.is_ok());
        let generation = generation.unwrap();
        
        // Check if correct number of agents was created
        assert_eq!(generation.agents.len(), settings.generation_size);
        
        // Check if each network has the correct layer structure
        for agent in &generation.agents {
            assert_eq!(
                agent.neural_network.layer_sizes, 
                settings.neural_network_layer_structure
            );
            assert_eq!(agent.fitness, 0.0);
        }
    }
    
    #[test]
    fn test_create_initial_with_invalid_settings() {
        // Create settings with invalid population size
        let invalid_settings = Settings::default().with_generation_size(0);
        
        // Attempt to create a generation should fail
        let result = Generation::create_initial(&invalid_settings);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_create_initial_with_custom_settings() {
        // Custom settings
        let custom_settings = Settings::default()
            .with_generation_size(50)
            .with_network_architecture(vec![147, 100, 50, 132]);
            
        let generation = Generation::create_initial(&custom_settings).unwrap();
        
        // Check if correct number of agents was created
        assert_eq!(generation.agents.len(), 50);
        
        // Check layer structure
        for agent in &generation.agents {
            assert_eq!(
                agent.neural_network.layer_sizes, 
                vec![147, 100, 50, 132]
            );
        }
    }
    
    #[test]
    fn test_generation_utility_methods() {
        let settings = Settings::default().with_generation_size(5);
        let mut generation = Generation::create_initial(&settings).unwrap();
        
        assert_eq!(generation.size(), 5);
        
        // Set fitness for all agents
        for (i, agent) in generation.agents.iter_mut().enumerate() {
            agent.fitness = (i + 1) as f64;
        }
        
        let network = generation.get_neural_network(2);
        assert!(network.is_some());
        assert_eq!(network.unwrap().layer_sizes, settings.neural_network_layer_structure);
        
        // Test get_network_mut
        let network_mut = generation.get_neural_network_mut(3);
        assert!(network_mut.is_some());
    }
    
    #[test]
    fn test_set_fitness_scores() {
        let settings = Settings::default().with_generation_size(3);
        let mut generation = Generation::create_initial(&settings).unwrap();
        
        // Set fitness scores
        let scores = vec![10.0, 5.0, 8.0];
        assert!(generation.set_fitness_scores(scores.clone()).is_ok());
        
        // Check if scores were set correctly
        for (i, agent) in generation.agents.iter().enumerate() {
            assert_eq!(agent.fitness, scores[i]);
        }
        
        // Try setting wrong number of scores
        let wrong_scores = vec![1.0, 2.0];
        assert!(generation.set_fitness_scores(wrong_scores).is_err());
    }
    
    #[test]
    fn test_sort_by_fitness() {
        let settings = Settings::default().with_generation_size(4);
        let mut generation = Generation::create_initial(&settings).unwrap();
        
        // Set fitness scores
        let scores = vec![3.0, 1.0, 4.0, 2.0];
        generation.set_fitness_scores(scores).unwrap();
        
        // Sort by fitness
        assert!(generation.sort_by_fitness().is_ok());
        
        // Check if sorted correctly (highest first)
        let sorted_fitness: Vec<f64> = generation.agents
            .iter()
            .map(|agent| agent.fitness)
            .collect();
        assert_eq!(sorted_fitness, vec![4.0, 3.0, 2.0, 1.0]);
    }
}

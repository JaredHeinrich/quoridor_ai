use neural_network::neural_network::NeuralNetwork;

#[derive(Debug, Clone)]
pub struct Agent {
    /// The neural network
    pub neural_network: NeuralNetwork,
    /// The agent's fitness score (if evaluated)
    pub fitness: Option<f64>,
}

impl Agent {
    /// Creates a new agent with the given neural network and zero fitness
    pub fn new(neural_network: NeuralNetwork) -> Self {
        Agent {
            neural_network,
            fitness: Some(0.0),
        }
    }

    /// Returns the agent's fitness score or 0.0 if not set
    pub fn get_fitness(&self) -> f64 {
        self.fitness.unwrap_or(0.0)
    }

    /// Increases the agent's fitness score by the given value
    /// If the fitness is not set, initializes it to the given value
    pub fn increase_fitness(&mut self, value: f64) {
        self.fitness = Some(match self.fitness {
            Some(current) => current + value,
            None => value,
        });
    }

    /// Sets the fitness score to a specific value
    pub fn set_fitness(&mut self, value: f64) {
        self.fitness = Some(value);
    }

    /// Resets the fitness score to zero
    pub fn reset_fitness(&mut self) {
        self.fitness = Some(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_agent() -> Agent {
        let nn = NeuralNetwork::new(&vec![10, 20, 5]).unwrap();
        Agent::new(nn)
    }

    #[test]
    fn test_new_agent_has_zero_fitness() {
        let agent = create_test_agent();
        assert_eq!(agent.get_fitness(), 0.0);
    }

    #[test]
    fn test_increase_fitness() {
        let mut agent = create_test_agent();
        
        // Increase by 10
        agent.increase_fitness(10.0);
        assert_eq!(agent.get_fitness(), 10.0);
        
        // Increase by another 5
        agent.increase_fitness(5.0);
        assert_eq!(agent.get_fitness(), 15.0);
    }

    #[test]
    fn test_set_fitness() {
        let mut agent = create_test_agent();
        
        agent.set_fitness(20.0);
        assert_eq!(agent.get_fitness(), 20.0);
        
        // Set to a different value
        agent.set_fitness(15.0);
        assert_eq!(agent.get_fitness(), 15.0);
    }

    #[test]
    fn test_reset_fitness() {
        let mut agent = create_test_agent();
        
        agent.set_fitness(30.0);
        agent.reset_fitness();
        assert_eq!(agent.get_fitness(), 0.0);
    }

    #[test]
    fn test_none_fitness_handling() {
        let mut agent = create_test_agent();
        agent.fitness = None;
        
        // Should return 0.0 if fitness is None
        assert_eq!(agent.get_fitness(), 0.0);
        
        // Should initialize fitness if None
        agent.increase_fitness(5.0);
        assert_eq!(agent.get_fitness(), 5.0);
    }
}
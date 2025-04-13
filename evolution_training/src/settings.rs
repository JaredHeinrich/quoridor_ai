use serde::{Deserialize, Serialize};
use crate::{error::EvolutionError, game_adapter::reward::RewardFunction};

/// Configuration settings for the evolutionary training process.
/// 
/// This struct contains all configurable parameters that control the behavior
/// of the evolutionary algorithm, including neural network structure, selection
/// parameters, reward function coefficients, and simulation constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    // Population parameters
    /// Number of neural networks in each generation
    pub generation_size: usize,
    
    /// Size of the input layer (number of input features)
    /// Note: This should always be 147 for Quoridor (81 board positions + 64 walls + 2 wall counts)
    pub input_layer_size: usize,

    /// Size of the output layer (number of output features)
    /// Note: This should always be 132 for Quoridor (4 directions + 64 horizontal walls + 64 vertical walls)
    pub output_layer_size: usize,

    // Neural network architecture
    /// Layer structure of neural networks [input_layer, hidden_layer(s), output_layer]
    /// Note: First layer must be input layer size (147) and last layer must be output layer size (132)
    pub neural_network_layer_structure: Vec<usize>,
    
    // Selection parameters
    /// Fraction of population that survives to next generation (range: 0.0-1.0)
    pub survival_rate: f64,

    /// Mutation rate for neural network weights and biases
    pub mutation_rate: f64,

    /// Percentual decrease of the mutation rate for each generation
    pub mutation_rate_decrease: f64,
    
    // Reward function coefficients
    pub reward_function: RewardFunction,
    /// Reward for winning the game
    pub win_reward: f64,
    /// Penalty based on agent's distance from goal (negative value)
    pub own_distance_punishment: f64,
    /// Reward based on opponent's distance from goal
    pub other_pawn_distance_reward: f64,
    /// Penalty for each turn taken (negative value to encourage efficiency)
    pub per_saved_turn_reward: f64,
    
    // Simulation constraints
    /// Maximum number of moves allowed per player before game termination
    pub max_moves_per_player: usize,
    /// Whether to select moves deterministically (highest output) or probabilistically
    pub play_deterministically: bool,
    /// Total number of generations to run
    pub number_of_generations: usize,
}

impl Default for Settings {
    /// Creates default settings with reasonable values for Quoridor AI training.
    fn default() -> Self {
        Self {
            // Default population of 100 neural networks per generation
            generation_size: 100,
            
            // Input layer size (147) and output layer size (132)
            input_layer_size: 147,
            output_layer_size: 132,

            // Default architecture with three hidden layers of 128 neurons each
            // Input layer (147) -> Hidden layer (128) -> Hidden layer (128) -> Hidden layer (128) -> Output layer (132)
            neural_network_layer_structure: vec![147, 128, 128, 128, 132],
            
            // 50% of population survives to next generation
            survival_rate: 0.5,
            // Medium mutation rate for diversity
            mutation_rate: 0.1,

            // Decrease mutation rate by 0.5% each generation
            mutation_rate_decrease: 0.005,
            
            // Reward 
            reward_function: RewardFunction::Simple,
            win_reward: 1000.0,
            own_distance_punishment: -10.0,
            other_pawn_distance_reward: 5.0,
            per_saved_turn_reward: 1.0,
            
            // Game constraints
            max_moves_per_player: 50,
            play_deterministically: true,
            number_of_generations: 1000,

        }
    }
}

impl Settings {
    /// Creates new settings with default values
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Validates that all settings are within acceptable ranges
    pub fn validate(&self) -> Result<(), EvolutionError> {
        // Population must be at least 2
        if self.generation_size < 2 {
            return Err(EvolutionError::InvalidPopulationSize(self.generation_size));
        }

        // Neural network structure validation
        if self.neural_network_layer_structure.len() < 2 {
            return Err(EvolutionError::TooFewLayers(self.neural_network_layer_structure.len()));
        }
        if self.neural_network_layer_structure[0] != self.input_layer_size {
            return Err(EvolutionError::InvalidInputLayerSize(self.input_layer_size, self.neural_network_layer_structure[0]));
        }
        if *self.neural_network_layer_structure.last().unwrap() != self.output_layer_size {
            return Err(EvolutionError::InvalidOutputLayerSize(self.output_layer_size, *self.neural_network_layer_structure.last().unwrap()));
        }
        if self.neural_network_layer_structure.iter().any(|&size| size == 0) {
            return Err(EvolutionError::EmptyLayer);
        }

        // Survival rate must be between 0 and 1
        if !(0.0..=1.0).contains(&self.survival_rate) {
            return Err(EvolutionError::InvalidSurvivalRate(self.survival_rate));
        }

        // Game constraints
        if self.max_moves_per_player == 0 {
            return Err(EvolutionError::InvalidMaxMoves);
        }
        if self.number_of_generations == 0 {
            return Err(EvolutionError::InvalidGenerationCount);
        }

        Ok(())
    }

    // Builder methods for fluent API

    /// Sets the population size (number of neural networks per generation)
    pub fn with_generation_size(mut self, size: usize) -> Self {
        self.generation_size = size;
        self
    }

    /// Sets the neural network layer structure
    /// Note: First layer must be 147 and last layer must be 132
    pub fn with_network_architecture(mut self, layers: Vec<usize>) -> Self {
        self.neural_network_layer_structure = layers;
        self
    }

    /// Sets the survival rate (fraction of population that survives to next generation)
    pub fn with_survival_rate(mut self, rate: f64) -> Self {
        self.survival_rate = rate;
        self
    }

    /// Sets the mutation rate for neural network weights and biases
    pub fn with_mutation_rate(mut self, rate: f64) -> Self {
        self.mutation_rate = rate;
        self
    }

    /// Sets all reward function coefficients at once
    pub fn with_reward_coefficients(
        mut self,
        function: RewardFunction,
        win_rew: f64,
        own_distance_pun: f64,
        other_distance_rew: f64,
        turn_rew: f64,
    ) -> Self {
        self.reward_function = function;
        self.win_reward = win_rew;
        self.own_distance_punishment = own_distance_pun;
        self.other_pawn_distance_reward = other_distance_rew;
        self.per_saved_turn_reward = turn_rew;
        self
    }

    /// Sets the maximum number of moves allowed per player
    pub fn with_max_moves_per_player(mut self, max_moves: usize) -> Self {
        self.max_moves_per_player = max_moves;
        self
    }

    /// Sets whether to play deterministically or probabilistically
    pub fn with_deterministic_play(mut self, deterministic: bool) -> Self {
        self.play_deterministically = deterministic;
        self
    }

    /// Sets the total number of generations to run
    pub fn with_generation_count(mut self, generations: usize) -> Self {
        self.number_of_generations = generations;
        self
    }

    // Helper methods to access derived information

    /// Returns the input layer size (always 147 for Quoridor)
    pub fn input_layer_size(&self) -> usize {
        self.neural_network_layer_structure[0]
    }

    /// Returns the output layer size (always 132 for Quoridor)
    pub fn output_layer_size(&self) -> usize {
        *self.neural_network_layer_structure.last().unwrap()
    }
    
    /// Returns the number of hidden layers
    pub fn hidden_layer_count(&self) -> usize {
        self.neural_network_layer_structure.len() - 2
    }
    
    /// Calculates how many networks survive to the next generation
    pub fn survivors_count(&self) -> usize {
        (self.generation_size as f64 * self.survival_rate).ceil() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validation() {
        // Valid settings should pass validation
        let settings = Settings::default();
        assert!(settings.validate().is_ok());
        
        // Invalid population size
        let settings = Settings::default().with_generation_size(1);
        assert!(settings.validate().is_err());
        
        // Invalid neural network structure (wrong input layer)
        let settings = Settings::default().with_network_architecture(vec![100, 50, 132]);
        assert!(settings.validate().is_err());
        
        // Invalid neural network structure (wrong output layer)
        let settings = Settings::default().with_network_architecture(vec![147, 50, 100]);
        assert!(settings.validate().is_err());
        
        // Invalid survival rate
        let settings = Settings::default().with_survival_rate(1.5);
        assert!(settings.validate().is_err());
    }
    
    #[test]
    fn test_builder_pattern() {
        let settings = Settings::new()
            .with_generation_size(200)
            .with_network_architecture(vec![147, 300, 200, 132])
            .with_survival_rate(0.3)
            .with_mutation_rate(0.05)
            .with_reward_coefficients(RewardFunction::Symmetric, 500.0, -5.0, 3.0, 0.5)
            .with_max_moves_per_player(150)
            .with_deterministic_play(true)
            .with_generation_count(500);
        
        assert_eq!(settings.generation_size, 200);
        assert_eq!(settings.neural_network_layer_structure, vec![147, 300, 200, 132]);
        assert_eq!(settings.survival_rate, 0.3);
        assert_eq!(settings.mutation_rate, 0.05);
        assert_eq!(settings.win_reward, 500.0);
        assert_eq!(settings.own_distance_punishment, -5.0);
        assert_eq!(settings.other_pawn_distance_reward, 3.0);
        assert_eq!(settings.per_saved_turn_reward, 0.5);
        assert_eq!(settings.max_moves_per_player, 150);
        assert_eq!(settings.play_deterministically, true);
        assert_eq!(settings.number_of_generations, 500);
    }
    
    #[test]
    fn test_helper_methods() {
        let settings = Settings::default()
            .with_generation_size(200)
            .with_network_architecture(vec![147, 64, 32, 132])
            .with_survival_rate(0.25);
        
        assert_eq!(settings.input_layer_size(), 147);
        assert_eq!(settings.output_layer_size(), 132);
        assert_eq!(settings.hidden_layer_count(), 2);
        assert_eq!(settings.survivors_count(), 50); // 25% of 200
    }
}
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EvolutionError {
    #[error("Population size must be at least 2, got {0}")]
    InvalidPopulationSize(usize),

    #[error("Neural network must have at least 2 layers")]
    TooFewLayers(usize),

    #[error("Input layer must have {0} neurons (81 board + 64 walls + 2 wall counts), got {1}")]
    InvalidInputLayerSize(usize, usize),

    #[error("Output layer must have {0} neurons (4 directions + 64*2 walls), got {1}")]
    InvalidOutputLayerSize(usize, usize),

    #[error("All neural network layers must have at least one neuron")]
    EmptyLayer,

    #[error("Survival rate must be between 0.0 and 1.0, got {0}")]
    InvalidSurvivalRate(f64),

    #[error("Maximum moves per player must be at least 1")]
    InvalidMaxMoves,

    #[error("Number of generations must be at least 1")]
    InvalidGenerationCount,

    #[error("Network initialization error: {0}")]
    NetworkError(#[from] neural_network::error::NNError),
}

#[derive(Error, Debug)]
pub enum GameAdapterError {
    #[error("Wall position out of bounds: ({0}, {1})")]
    WallPositionOutOfBounds(usize, usize),
    
    #[error("Invalid game state encoding")]
    InvalidEncoding,
    
    #[error("Invalid move decoding")]
    InvalidMoveDecoding,
}


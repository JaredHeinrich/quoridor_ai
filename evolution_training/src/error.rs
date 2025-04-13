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

    #[error("Reactivation rate must be between 0.0 and 1.0, got {0}")]
    InvalidReactivationRate(f64),

    #[error("Sum of Survival rate: {0} and reactivation rate: {1} must be between 0.0 and 1.0")]
    InvalidSurvivalAndReactivationRate(f64, f64),

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
    
    #[error("Invalid move decoding, output matrix: Rows: {0}, Columns: {1}")]
    InvalidMoveDecoding(usize, usize),

    #[error("No valid moves found, should never happen, as the game is designed to not enter a state with only invalid moves")]
    NoValidMoves,

    #[error("To use non-deterministic function, all values need to be probabilities >= 0. Value: {0}")]
    InvalidProbabilitySoftmax(f64),

    #[error("To use non-deterministic function, all values need to be probabilities >= 0 that add up to 1. Probabilities add up to {0}")]
    InvalidProbabilitiesSumSoftmax(f64),

    #[error("Index out of bounds for possible Moves")]
    IndexMovesOutOfBounds,

    #[error("Invalid move")]
    InvalidMove
}   


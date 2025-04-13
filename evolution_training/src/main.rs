pub mod training_environment;
pub mod settings;
pub mod error;

pub mod evolution {
    pub mod generation;
    pub mod selection;
    pub mod lib;
}

pub mod game_adapter {
    pub mod board_encoder;
    pub mod move_decoder;
    pub mod reward;
    pub mod lib;
}

fn main() {
    
}

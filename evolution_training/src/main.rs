pub mod training_environment;
pub mod settings;
pub mod logging;

pub mod evolution {
    pub mod generation;
    pub mod selection;
    pub mod r#mod;
}

pub mod game_adapter {
    pub mod board_encoder;
    pub mod move_decoder;
    pub mod reward;
    pub mod mod;
}

fn main() {
    
}

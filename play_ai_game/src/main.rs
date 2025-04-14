use neural_network_logger::{logger, models::LogEntry};
use quoridor::game_state::Game;
use neural_network::neural_network::NeuralNetwork;
use bevy::prelude::*;


fn main() {
    const BOARD_SIZE: i16 = 9;
    const WALLS_PER_PLAYER: i16 = 10;
    let game_state: Game = Game::new(BOARD_SIZE, WALLS_PER_PLAYER);
    let mut ai_players: Vec<LogEntry> = logger::read_specific_lines(&[0, 1], "../evolution_training/evolution_history.json").unwrap();
    let ai_players: [NeuralNetwork; 2] = [ai_players.pop().unwrap().neural_network, ai_players.pop().unwrap().neural_network];
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    render::board::spawn_board(&mut commands, &mut meshes, &mut materials);
}

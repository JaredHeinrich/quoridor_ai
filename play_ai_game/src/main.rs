use bevy::{
    color::palettes::css::{GREEN, RED},
    prelude::*,
};
use evolution_training::{settings::Settings, training_environment::TrainingEnvironment};
use neural_network::neural_network::NeuralNetwork;
use neural_network_logger::{logger, models::LogEntry};
use quoridor::{game_state::Game, pawn::Pawn};

const PAWN_ONE_COLOR: Srgba = GREEN;
const PAWN_TWO_COLOR: Srgba = RED;

#[derive(Resource)]
struct GameState(Game);

#[derive(Resource)]
struct AIPlayers([NeuralNetwork; 2]);

#[derive(Resource)]
struct Env(TrainingEnvironment);

fn main() {
    const BOARD_SIZE: i16 = 9;
    const WALLS_PER_PLAYER: i16 = 10;

    let game_state: Game = Game::new(BOARD_SIZE, WALLS_PER_PLAYER);
    let mut ai_players: Vec<LogEntry> =
        logger::read_specific_lines(&[19981, 19982], "../evolution_training/evolution_history_default_1000_iterations.json")
            .unwrap();
    let ai_players: [NeuralNetwork; 2] = [
        ai_players.pop().unwrap().neural_network,
        ai_players.pop().unwrap().neural_network,
    ];
    let settings: Settings = Settings::new();
    let training_environment: TrainingEnvironment = TrainingEnvironment::new(settings);
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(GameState(game_state))
        .insert_resource(AIPlayers(ai_players))
        .insert_resource(Env(training_environment))
        .add_systems(Startup, setup)
        .add_systems(Update, handle_key_input)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut pawns: Query<Entity, With<render::pawn::Pawn>>,
    game_state: Res<GameState>,
) {
    commands.spawn(Camera2d);
    render::board::spawn_board(&mut commands, &mut meshes, &mut materials);
    update_pawns(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut pawns,
        game_state.as_ref(),
    );
}

fn update_pawns(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    pawns: &mut Query<Entity, With<render::pawn::Pawn>>,
    game_state: &GameState,
) {
    render::pawn::clear_pawns(commands, pawns);
    let pawns: &[Pawn; 2] = &game_state.0.pawns;
    render::pawn::spawn_pawn(
        commands,
        meshes,
        materials,
        pawns[0].position,
        PAWN_ONE_COLOR,
    );
    render::pawn::spawn_pawn(
        commands,
        meshes,
        materials,
        pawns[1].position,
        PAWN_TWO_COLOR,
    );
}

fn update_walls(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    walls: &mut Query<Entity, With<render::wall::Wall>>,
    game_state: &GameState,
) {
    render::wall::clear_walls(commands, walls);
    let game_walls = &game_state.0.walls;
    for wall in game_walls {
        render::wall::spawn_wall(commands, meshes, materials, wall.position, wall.orientation);
    }
}

fn handle_key_input(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut pawns: Query<Entity, With<render::pawn::Pawn>>,
    mut walls: Query<Entity, With<render::wall::Wall>>,
    mut game_state: ResMut<GameState>,
    ai_players: Res<AIPlayers>,
    environment: Res<Env>,
) {
    if keyboard_input.just_pressed(KeyCode::Enter) {
        let current_pawn = game_state.0.current_pawn;

        environment.0.nn_move(
            &ai_players.0[current_pawn],
            &mut game_state.0,
            0,
            neural_network::neural_network::OutputActivation::Sigmoid,
        );
        update_pawns(
            &mut commands,
            &mut meshes,
            &mut materials,
            &mut pawns,
            game_state.as_ref(),
        );
        update_walls(
            &mut commands,
            &mut meshes,
            &mut materials,
            &mut walls,
            game_state.as_ref(),
        );
        println!("Enter Pressed");
    }
}

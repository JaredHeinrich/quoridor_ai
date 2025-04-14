use crate::game_adapter::board_encoder::distance_to_goal;
use crate::game_adapter::board_encoder::encode_board;
use crate::game_adapter::move_decoder::decode_move;
use crate::settings::Settings;
use anyhow::Result;
use neural_network::neural_network::{NeuralNetwork, OutputActivation};
use quoridor::direction::Direction;
use quoridor::game_state::{Game, Move};
use quoridor::vector::Vector;
use quoridor::vector::VectorUtility; // Add this line
use quoridor::wall::{Orientation, Wall};
use rand::Rng;

/// Helper function to get all valid wall placements for a game
fn get_valid_walls(game: &Game) -> Vec<Wall> {
    let mut valid_walls = Vec::new();

    // Skip if player has no walls left
    if game.pawns[game.current_pawn].number_of_available_walls <= 0 {
        return valid_walls;
    }

    // Wall grid is one size smaller than the board grid
    let wall_grid_size = game.board_size as usize - 1;

    // Try all possible wall positions and orientations
    for y in 0..wall_grid_size {
        for x in 0..wall_grid_size {
            let position = Vector::new(x as i16, y as i16);

            // Try horizontal wall
            let h_wall = Wall::new(position, Orientation::Horizontal);
            if game.is_wall_valid(&h_wall) {
                valid_walls.push(h_wall);
            }

            // Try vertical wall
            let v_wall = Wall::new(position, Orientation::Vertical);
            if game.is_wall_valid(&v_wall) {
                valid_walls.push(v_wall);
            }
        }
    }

    valid_walls
}

/// Trait defining the interface for benchmark agents
pub trait BenchmarkAgent {
    /// Generate a move for the current game state
    fn generate_move(&self, game: &Game) -> Result<Move>;

    /// Get the name of the agent for display purposes
    fn name(&self) -> &str;
}

/// Agent that makes random valid moves
pub struct RandomAgent;

impl RandomAgent {
    pub fn new() -> Self {
        RandomAgent
    }
}

impl BenchmarkAgent for RandomAgent {
    fn generate_move(&self, game: &Game) -> Result<Move> {
        // Get all valid pawn moves
        let valid_pawn_positions = game.valid_next_positions();
        let mut all_valid_moves = Vec::new();

        // Add pawn moves
        for pos in valid_pawn_positions {
            all_valid_moves.push(Move::PawnMove(pos));
        }

        // Add wall moves if player has walls left
        if game.pawns[game.current_pawn].number_of_available_walls > 0 {
            // Get valid wall placements using our helper function
            let valid_walls = get_valid_walls(game);
            for wall in valid_walls {
                all_valid_moves.push(Move::WallMove(wall));
            }
        }

        if all_valid_moves.is_empty() {
            return Err(anyhow::anyhow!("No valid moves available"));
        }

        // Select a random move
        let mut rng = rand::rng();
        let random_index = rng.random_range(0..all_valid_moves.len());
        Ok(all_valid_moves[random_index].clone())
    }

    fn name(&self) -> &str {
        "Random Agent"
    }
}

/// Agent that always tries to move forward toward the goal
pub struct SimpleForwardAgent;

impl SimpleForwardAgent {
    pub fn new() -> Self {
        SimpleForwardAgent
    }
}

impl BenchmarkAgent for SimpleForwardAgent {
    fn generate_move(&self, game: &Game) -> Result<Move> {
        let current_player = game.current_pawn;
        let current_position = game.pawns[current_player].position;
        let goal_line = game.pawns[current_player].goal_line;

        // Determine forward direction (toward goal)
        let forward_direction = if goal_line < current_position.y {
            Direction::Up
        } else {
            Direction::Down
        };

        let valid_positions = game.valid_next_positions();

        // First priority: Try to move forward if possible
        let forward_pos = current_position.add(forward_direction.to_vector());
        if valid_positions.contains(&forward_pos) {
            return Ok(Move::PawnMove(forward_pos));
        }

        // Second priority: Try to move sideways if forward is blocked
        let left_direction = forward_direction.turn_left();
        let right_direction = forward_direction.turn_right();

        let left_pos = current_position.add(left_direction.to_vector());
        let right_pos = current_position.add(right_direction.to_vector());

        if valid_positions.contains(&left_pos) {
            return Ok(Move::PawnMove(left_pos));
        } else if valid_positions.contains(&right_pos) {
            return Ok(Move::PawnMove(right_pos));
        }

        // Third priority: Just pick any valid pawn move
        if !valid_positions.is_empty() {
            return Ok(Move::PawnMove(valid_positions[0]));
        }

        // Last resort: Try placing a wall
        if game.pawns[current_player].number_of_available_walls > 0 {
            let valid_walls = get_valid_walls(game);
            if !valid_walls.is_empty() {
                return Ok(Move::WallMove(valid_walls[0].clone()));
            }
        }

        Err(anyhow::anyhow!("No valid moves available"))
    }

    fn name(&self) -> &str {
        "Forward Agent"
    }
}

/// Play a game between a neural network and a benchmark agent
pub fn play_against_benchmark(
    neural_network: &NeuralNetwork,
    benchmark_agent: &dyn BenchmarkAgent,
    settings: &Settings,
    neural_network_plays_first: bool,
) -> Result<(f64, f64)> {
    let mut game = Game::new(settings.board_size as i16, settings.walls_per_player as i16);

    // Determine which player is controlled by the neural network
    let nn_player_index = if neural_network_plays_first { 0 } else { 1 };

    // Play until someone wins or max moves reached
    for _move_counter in 0..settings.max_moves_per_player * 2 {
        let current_player = game.current_pawn;

        // Determine whose turn it is and get the move
        let game_move = if current_player == nn_player_index {
            // Neural network's turn
            let game_state = encode_board(&game)?;
            let output_activation = if settings.play_deterministically {
                OutputActivation::Sigmoid
            } else {
                OutputActivation::Softmax
            };

            let nn_output = neural_network.feed_forward(game_state, output_activation)?;
            decode_move(&nn_output, &game, settings)?
        } else {
            // Benchmark agent's turn
            benchmark_agent.generate_move(&game)?
        };

        // Execute the move
        if let Err(e) = game.make_move(game_move.clone()) {
            return Err(anyhow::anyhow!("Invalid move: {}", e));
        }

        // Check if someone has reached their goal
        if distance_to_goal(&game, 0) == 0 || distance_to_goal(&game, 1) == 0 {
            let winner = if distance_to_goal(&game, 0) == 0 {
                0
            } else {
                1
            };

            // Calculate scores (1.0 for win, 0.0 for loss)
            let nn_score = if winner == nn_player_index { 1.0 } else { 0.0 };
            let benchmark_score = 1.0 - nn_score;

            return Ok((nn_score, benchmark_score));
        }
    }

    // Draw if max moves reached without a winner
    Ok((0.5, 0.5))
}

#[cfg(test)]
mod tests {
    use super::*;
    use neural_network::neural_network::NeuralNetwork;

    #[test]
    fn test_random_agent_generates_valid_move() -> Result<()> {
        let game = Game::new(9, 10);
        let agent = RandomAgent::new();

        let game_move = agent.generate_move(&game)?;

        // Verify the move is valid by applying it
        let mut game_copy = game.clone();
        assert!(game_copy.make_move(game_move).is_ok());

        Ok(())
    }

    #[test]
    fn test_forward_agent_prefers_forward_movement() -> Result<()> {
        let game = Game::new(9, 10);
        let agent = SimpleForwardAgent::new();

        let game_move = agent.generate_move(&game)?;

        // First player starts at (4, 8) and should move to (4, 7)
        match game_move {
            Move::PawnMove(pos) => {
                assert_eq!(pos.x, 4);
                assert_eq!(pos.y, 7);
            }
            _ => panic!("Expected pawn move, got wall move"),
        }

        Ok(())
    }

    #[test]
    fn test_forward_agent_handles_blocked_path() -> Result<()> {
        // Create a game with a wall blocking the forward path
        let mut game = Game::new(9, 10);
        let wall =
            quoridor::wall::Wall::new(Vector::new(3, 7), quoridor::wall::Orientation::Horizontal);
        game.walls.push(wall);

        let agent = SimpleForwardAgent::new();
        let game_move = agent.generate_move(&game)?;

        // Agent should move sideways since forward is blocked
        match game_move {
            Move::PawnMove(pos) => {
                assert!(pos.x == 3 || pos.x == 5); // Left or right
                assert_eq!(pos.y, 8); // Same y
            }
            _ => panic!("Expected pawn move, got wall move"),
        }

        Ok(())
    }

    #[test]
    fn test_play_against_benchmark() -> Result<()> {
        let settings = Settings::default()
            .with_max_moves_per_player(50)
            .with_deterministic_play(true);

        // Create neural network with sizes from settings
        let nn = NeuralNetwork::new(&vec![
            settings.input_layer_size,  // Get input size from settings
            64,                         // Hidden layer
            settings.output_layer_size, // Get output size from settings
        ])?;

        let agent = RandomAgent::new();

        let (nn_score, agent_score) = play_against_benchmark(&nn, &agent, &settings, true)?;

        // Verify scores sum to 1.0 (win/loss/draw)
        assert!((nn_score + agent_score - 1.0).abs() < 1e-6);

        Ok(())
    }
}

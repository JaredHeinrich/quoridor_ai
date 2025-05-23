use anyhow::Result;
use matrix::matrix::Matrix;
use quoridor::direction::Direction;
use quoridor::game_state::{Game, Move};
use quoridor::vector::{Vector, VectorUtility};
use quoridor::wall::{Orientation, Wall};
use rand::{rng, Rng};

use crate::error::GameAdapterError;
use crate::settings::Settings;

/// Decodes neural network output into a valid game move.
/// The output is in the perspective of the current player. It needs to be transferred to the game's perspective first.
///
/// The neural network output has 132 values representing:
/// - First 4 values: Pawn movements (UP, RIGHT, DOWN, LEFT)
/// - Next 64 values: Horizontal wall placements (row by row)
/// - Last 64 values: Vertical wall placements (row by row)
///
/// # Arguments
/// * `nn_output` - 132x1 Matrix containing the neural network's output
/// * `game` - Current game state
/// * `play_deterministically` - If true, choose the highest-valued valid move;
///   if false, use output values as weights for probabilistic move selection
///
/// # Returns
/// A valid `Move` (from the game perspective) that can be applied to the game
pub fn decode_move(nn_output: &mut Matrix, game: &Game, settings: &Settings) -> Result<Move> {
    // Validate output matrix dimensions
    if nn_output.rows != settings.output_layer_size || nn_output.columns != 1 {
        return Err(
            GameAdapterError::InvalidMoveDecoding(nn_output.rows, nn_output.columns).into(),
        );
    }

    if game.current_pawn == 1 {
        // If it is the second's players turn the output_matrix needs to be interpreted differently so that the game understands it
        change_output_layer_to_game_perspective(nn_output, game);
    }

    if settings.play_deterministically {
        select_deterministic_move(nn_output, game)
    } else {
        select_probabilistic_move(nn_output, game)
    }
}

fn change_output_layer_to_game_perspective(nn_output: &mut Matrix, game: &Game) {
    if game.current_pawn == 1 {
        // Swap pawn movement directions (rotate 180°)
        // 0 UP <-> 2 DOWN
        nn_output.values.swap(0, 2);

        // 1 RIGHT <-> 3 LEFT
        nn_output.values.swap(1, 3);

        // Reverse horizontal walls (indices 4-67)
        reverse_segment(&mut nn_output.values, 4, 67);

        // Reverse vertical walls (indices 68-131)
        reverse_segment(&mut nn_output.values, 68, 131);
    }
}

/// Helper function to reverse a segment of values in-place
fn reverse_segment(values: &mut [f64], start: usize, end: usize) {
    let mut left = start;
    let mut right = end;
    while left < right {
        values.swap(left, right);
        left += 1;
        right -= 1;
    }
}

/// Selects the highest-valued valid move
fn select_deterministic_move(nn_output: &Matrix, game: &Game) -> Result<Move> {
    // Keep track of which indices we've already checked
    let mut checked_indices = vec![false; nn_output.values.len()];
    let mut unchecked_count = nn_output.values.len();

    // Continue until all indices are checked
    while unchecked_count > 0 {
        // Select the index with highest value from the neural network output
        let mut max_index = 0;
        let mut max_value = f64::NEG_INFINITY;

        for (i, &value) in nn_output.values.iter().enumerate() {
            if !checked_indices[i] && value > max_value {
                max_value = value;
                max_index = i;
            }
        }

        // Mark this index as checked
        checked_indices[max_index] = true;
        unchecked_count -= 1;

        // Check if this is a valid move, if yes return
        let created_move = try_create_move(max_index, game);
        if created_move.is_ok() {
            // If the move is valid, return it
            return created_move;
        }
        // If not a valid move, loop continues to check the next highest value left
    }

    // No valid moves found
    Err(GameAdapterError::NoValidMoves.into())
}

/// Selects a move probabilistically based on output values
/// Only use if the output matrix used softmax as activation function
fn select_probabilistic_move(nn_output: &Matrix, game: &Game) -> Result<Move> {
    // should only be used if softmax was used previously as activation function

    // Check at least that no output value is negative
    if let Some(&val) = nn_output.values.iter().find(|&&v| v < 0.0) {
        return Err(GameAdapterError::InvalidProbabilitySoftmax(val).into());
    }

    // After collecting valid moves
    let mut valid_moves: Vec<(Move, f64)> = Vec::new();

    // Try each possible move
    for index in 0..132 {
        if let Ok(game_move) = try_create_move(index, game) {
            valid_moves.push((game_move, *nn_output.value(index, 0)));
        }
    }

    if valid_moves.is_empty() {
        return Err(GameAdapterError::NoValidMoves.into());
    }

    // Renormalize probabilities so they sum to 1.0, necessary since only valid moves considered
    let sum: f64 = valid_moves.iter().map(|(_, p)| *p).sum();
    if sum > 0.0 {
        for (_, prob) in &mut valid_moves {
            *prob /= sum;
        }
    }

    // Extract moves and weights
    let moves: Vec<Move> = valid_moves.iter().map(|(m, _)| m.clone()).collect();
    let probabilities: Vec<f64> = valid_moves.iter().map(|(_, p)| *p).collect();

    // Calculate the sum of probabilities
    let sum_probabilities: f64 = probabilities.iter().sum();
    // Sum should be close to 1.0
    if sum_probabilities >= 1.0 + 1e-6 || sum_probabilities <= 1.0 - 1e-6 {
        return Err(GameAdapterError::InvalidProbabilitiesSumSoftmax(sum_probabilities).into());
    }

    // Choose a move based on weighted probability
    let mut rng = rng();
    let random_value: f64 = rng.random_range(0.0..sum_probabilities);

    let mut cumulative_prob = 0.0;
    for i in 0..probabilities.len() {
        cumulative_prob += probabilities[i];
        if random_value <= cumulative_prob {
            return Ok(moves[i].clone());
        }
    }
    // no move chosen
    return Err(GameAdapterError::InvalidProbabilitiesSumSoftmax(sum_probabilities).into());
}

/// Attempts to create a valid move from the neural network output index
fn try_create_move(index: usize, game: &Game) -> Result<Move> {
    let board_size: usize = 9;
    let wall_size: usize = board_size - 1;
    if index < 4 {
        // Pawn move (first 4 values)
        try_create_pawn_move(index, game)
    } else if index < 4 + wall_size * wall_size {
        // Horizontal wall (next 64 values)
        let wall_index = index - 4;
        try_create_wall_move(wall_index, Orientation::Horizontal, game)
    } else if index < 4 + 2 * wall_size * wall_size {
        // Vertical wall (last 64 values)
        let wall_index = index - 4 - wall_size * wall_size;
        try_create_wall_move(wall_index, Orientation::Vertical, game)
    } else {
        Err(GameAdapterError::IndexMovesOutOfBounds.into())
    }
}

/// Attempts to create a valid pawn move
fn try_create_pawn_move(direction_index: usize, game: &Game) -> Result<Move> {
    let direction = match direction_index {
        0 => Direction::Up,
        1 => Direction::Right,
        2 => Direction::Down,
        3 => Direction::Left,
        _ => return Err(GameAdapterError::IndexMovesOutOfBounds.into()),
    };

    let current_position = game.current_pawn().position;

    let valid_positions = game.valid_next_positions();

    // Calculate the expected new position based on direction
    let movement = direction.to_vector();
    let expected_position = current_position.add(movement);

    // Check if the expected position is valid
    if valid_positions.contains(&expected_position) {
        return Ok(Move::PawnMove(expected_position));
    }

    // Handle special case: jumping over opponent
    let opponent_position = game.other_pawn().position;
    if expected_position == opponent_position {
        // Try to find a valid jump position (in the same direction)
        let jump_position = opponent_position.add(movement);
        if valid_positions.contains(&jump_position) {
            return Ok(Move::PawnMove(jump_position));
        }

        // Try diagonal jumps if straight jump is not possible
        let left_direction = direction.turn_left();
        let right_direction = direction.turn_right();
        let left_jump = opponent_position.add(left_direction.to_vector());
        let right_jump = opponent_position.add(right_direction.to_vector());

        if valid_positions.contains(&left_jump) {
            return Ok(Move::PawnMove(left_jump));
        } else if valid_positions.contains(&right_jump) {
            return Ok(Move::PawnMove(right_jump));
        }
    }

    Err(GameAdapterError::InvalidMove.into())
}

/// Attempts to create a valid wall move
fn try_create_wall_move(wall_index: usize, orientation: Orientation, game: &Game) -> Result<Move> {
    //If no walls left definetly Invalid Move
    if game.current_pawn().number_of_available_walls <= 0 {
        return Err(GameAdapterError::InvalidMove.into());
    }

    // Wall grid is 8x8 (for 9x9 board)
    let wall_grid_size = game.board_size as usize - 1;
    if wall_index >= wall_grid_size * wall_grid_size {
        return Err(GameAdapterError::IndexMovesOutOfBounds.into());
    }

    // Calculate wall position
    let x = wall_index % wall_grid_size;
    let y = wall_index / wall_grid_size;

    let wall = Wall::new(Vector::new(x as i16, y as i16), orientation);

    // Check if wall placement is valid
    if game.is_wall_valid(&wall) {
        Ok(Move::WallMove(wall))
    } else {
        Err(GameAdapterError::InvalidMove.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quoridor::pawn::Pawn;

    #[test]
    fn test_deterministic_pawn_move() {
        // Create a game with default setup
        let game = Game::new(9, 10);
        let settings = &Settings::default().with_deterministic_play(true);

        // Create output that strongly prefers UP direction (index 0)
        let mut values = vec![0.0; 132];
        values[0] = 10.0; // UP - should be valid for player at bottom

        let mut nn_output = Matrix::new(132, 1, values).unwrap();

        let result = decode_move(&mut nn_output, &game, settings).unwrap();

        match result {
            Move::PawnMove(pos) => {
                // Starting position is (4, 8), moving up should be (4, 7)
                assert_eq!(pos.x, 4);
                assert_eq!(pos.y, 7);
            }
            _ => panic!("Expected pawn move, got wall move"),
        }
    }

    #[test]
    fn test_deterministic_fallback_to_next_highest() {
        // Create a game with a wall blocking the UP direction
        let mut game = Game::new(9, 10);
        let settings = &Settings::default().with_deterministic_play(true);
        game.walls
            .push(Wall::new(Vector::new(4, 7), Orientation::Horizontal));

        // Create output that prefers UP, then RIGHT
        let mut values = vec![0.0; 132];
        values[0] = 10.0; // UP (blocked)
        values[1] = 8.0; // RIGHT (should be chosen)

        let mut nn_output = Matrix::new(132, 1, values).unwrap();

        let result = decode_move(&mut nn_output, &game, settings).unwrap();

        match result {
            Move::PawnMove(pos) => {
                // Starting position is (4, 8), moving right should be (5, 8)
                assert_eq!(pos.x, 5);
                assert_eq!(pos.y, 8);
            }
            _ => panic!("Expected pawn move, got wall move"),
        }
    }

    #[test]
    fn test_deterministic_wall_placement() {
        let game = Game::new(9, 10);
        let settings = &Settings::default().with_deterministic_play(true);

        // Create output that strongly prefers a horizontal wall at position (3, 3)
        let mut values = vec![0.0; 132];
        let wall_index = 4 + 3 + (3 * 8); // 4 (pawn moves) + 3 + (3 * 8) = 31
        values[wall_index] = 10.0;

        let mut nn_output = Matrix::new(132, 1, values).unwrap();

        let result = decode_move(&mut nn_output, &game, settings).unwrap();

        match result {
            Move::WallMove(wall) => {
                assert_eq!(wall.position.x, 3);
                assert_eq!(wall.position.y, 3);
                assert_eq!(wall.orientation, Orientation::Horizontal);
            }
            _ => panic!("Expected wall move, got pawn move"),
        }
    }

    #[test]
    fn test_probabilistic_move_selection() {
        let game = Game::new(9, 10);
        let settings = &Settings::default().with_deterministic_play(false);

        // Create relatively uniform output (slightly prefer pawn moves)
        let mut values = vec![1.0 / 128.0 / 3.0; 132];

        for i in 0..4 {
            values[i] = 1.0 / 6.0;
        }

        let mut nn_output = Matrix::new(132, 1, values).unwrap();

        // Test multiple outcomes to verify probabilistic behavior
        let mut pawn_moves = 0;
        let mut wall_moves = 0;

        for _ in 0..100 {
            let result = decode_move(&mut nn_output, &game, settings).unwrap();
            match result {
                Move::PawnMove(_) => pawn_moves += 1,
                Move::WallMove(_) => wall_moves += 1,
            }
        }

        // but still a significant number of wall moves
        assert!(pawn_moves > 0);
        assert!(wall_moves > 0);
        // With our setup, pawn moves should be more common
        assert!(pawn_moves > wall_moves);
    }

    #[test]
    fn test_jump_over_opponent() {
        let settings = &Settings::default().with_deterministic_play(true);
        // Create a game with pawns next to each other
        let board_size = 9;
        let pawns = [
            Pawn {
                position: Vector::new(4, 4),
                number_of_available_walls: 10,
                goal_line: 0,
            },
            Pawn {
                position: Vector::new(4, 3), // Directly above first pawn
                number_of_available_walls: 10,
                goal_line: 8,
            },
        ];

        let game = Game {
            pawns,
            current_pawn: 0,
            board_size,
            walls: Vec::new(),
        };

        // Create output that strongly prefers UP direction
        let mut values = vec![0.0; 132];
        values[0] = 10.0; // UP

        let mut nn_output = Matrix::new(132, 1, values).unwrap();

        let result = decode_move(&mut nn_output, &game, settings).unwrap();

        match result {
            Move::PawnMove(pos) => {
                // Should jump to (4, 2) - two spaces up
                assert_eq!(pos.x, 4);
                assert_eq!(pos.y, 2);
            }
            _ => panic!("Expected pawn move, got wall move"),
        }
    }

    #[test]
    fn test_invalid_output_size() {
        let game = Game::new(9, 10);
        let settings = &Settings::default().with_deterministic_play(true);

        // Create output with wrong size
        let mut nn_output = Matrix::new(100, 1, vec![0.0; 100]).unwrap();

        let result = decode_move(&mut nn_output, &game, settings);
        assert!(result.is_err());
    }

    #[test]
    fn test_change_output_layer_to_game_perspective() {
        let mut game = Game::new(9, 10);

        // Create a sample neural network output
        let mut values = vec![0.0; 132];
        values[0] = 1.0; // UP
        values[1] = 2.0; // RIGHT
        values[2] = 3.0; // DOWN
        values[3] = 4.0; // LEFT

        // Set some horizontal and vertical wall values
        values[4] = 5.0; // First horizontal wall
        values[66] = 6.0; // Second last horizontal wall]
        values[67] = 7.0; // Last horizontal wall
        values[68] = 8.0; // First vertical wall
        values[131] = 9.0; // Last vertical wall

        let mut nn_output = Matrix::new(132, 1, values).unwrap();

        // For player 0, no change should occur
        let mut nn_output_p0 = nn_output.clone();
        change_output_layer_to_game_perspective(&mut nn_output_p0, &game);

        assert_eq!(nn_output_p0.values[0], 1.0); // UP remains UP
        assert_eq!(nn_output_p0.values[1], 2.0); // RIGHT remains RIGHT

        // For player 1, changes should occur
        game.current_pawn = 1;
        change_output_layer_to_game_perspective(&mut nn_output, &game);

        assert_eq!(nn_output.values[0], 3.0); // UP becomes DOWN
        assert_eq!(nn_output.values[1], 4.0); // RIGHT becomes LEFT
        assert_eq!(nn_output.values[2], 1.0); // DOWN becomes UP
        assert_eq!(nn_output.values[3], 2.0); // LEFT becomes RIGHT
        assert_eq!(nn_output.values[4], 7.0); // First horizontal wall becomes last
        assert_eq!(nn_output.values[5], 6.0); // Second last horizontal wall becomes second
        assert_eq!(nn_output.values[67], 5.0); // Last horizontal wall becomes first
        assert_eq!(nn_output.values[68], 9.0); // First vertical wall becomes last
        assert_eq!(nn_output.values[131], 8.0); // Last vertical wall becomes first
    }

    #[test]
    fn test_change_output_layer_with_complex_pattern() {
        let mut game = Game::new(9, 10);
        game.current_pawn = 1;

        // Create a sample neural network output with sequential values
        let mut values = vec![0.0; 132];

        // Set a pattern for horizontal walls (indices 4-67)
        for i in 0..64 {
            values[4 + i] = i as f64;
        }

        // Set a pattern for vertical walls (indices 68-131)
        for i in 0..64 {
            values[68 + i] = 100.0 + i as f64;
        }

        let mut nn_output = Matrix::new(132, 1, values).unwrap();

        // Apply the perspective change
        change_output_layer_to_game_perspective(&mut nn_output, &game);

        // Verify horizontal walls are reversed
        for i in 0..64 {
            assert_eq!(nn_output.values[4 + i], (63 - i) as f64);
        }

        // Verify vertical walls are reversed
        for i in 0..64 {
            assert_eq!(nn_output.values[68 + i], 100.0 + (63 - i) as f64);
        }
    }
}

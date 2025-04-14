use crate::error::GameAdapterError;
use anyhow::Result;
use matrix::matrix::Matrix;
use quoridor::game_state::Game;
use quoridor::wall::Orientation;

/// Encodes a Quoridor game state into a neural network input representation.
///
/// The encoding follows this format:
/// 1. 81 values (9x9 board) representing pawn positions:
///    * 1.0: current player's pawn
///    * -1.0: opponent's pawn
///    * 0.0: no pawn
/// 2. 64 values (8x8 grid) representing walls:
///    * 1.0: horizontal wall
///    * -1.0: vertical wall
///    * 0.0: no wall
/// 3. 2 values representing wall counts:
///    * Current player's remaining walls
///    * Opponent's remaining walls
///
/// Total input size: 81 + 64 + 2 = 147 values
///
/// The board is always oriented from the perspective of the specified player,
/// ensuring consistent encoding regardless of which player the neural network is playing as.
///
/// # Arguments
/// * `game` - The current game state
/// * `player_perspective` - The perspective from which to encode (0 or 1)
///
/// # Returns
/// A 147x1 Matrix containing the encoded board state
pub fn encode_board(game: &Game) -> Result<Matrix> {
    let board_size = game.board_size as usize;
    let wall_grid_size = board_size - 1;
    let input_layer_size = board_size * board_size + wall_grid_size * wall_grid_size + 2;

    let mut values = Vec::with_capacity(input_layer_size);

    // Encode pawn positions (81 values for 9x9 board)
    encode_pawn_positions(game, &mut values)?;

    // Encode wall positions (64 values for 8x8 wall grid)
    encode_wall_positions(game, &mut values)?;

    // Encode wall counts (2 values)
    encode_wall_counts(game, &mut values);

    // Create matrix (column vector)
    Matrix::new(input_layer_size, 1, values)
}

/// Encodes pawn positions from the perspective of the given player
fn encode_pawn_positions(game: &Game, values: &mut Vec<f64>) -> Result<()> {
    let board_size = game.board_size as usize;
    let own_pawn = game.current_pawn();
    let opponent_pawn = game.other_pawn();

    // Initialize all positions to 0.0 (empty)
    let mut board = vec![vec![0.0; board_size]; board_size];

    // Mark the pawns based on player perspective
    let (own_x, own_y) = (own_pawn.position.x as usize, own_pawn.position.y as usize);
    let (opp_x, opp_y) = (
        opponent_pawn.position.x as usize,
        opponent_pawn.position.y as usize,
    );

    // Transform coordinates based on player_perspective
    let (own_x_transformed, own_y_transformed) =
        transform_coordinates(own_x, own_y, game.current_pawn, board_size);
    let (opp_x_transformed, opp_y_transformed) =
        transform_coordinates(opp_x, opp_y, game.current_pawn, board_size);

    // Set pawn positions in transformed coordinates rust syntax [row][col]
    board[own_y_transformed][own_x_transformed] = 1.0; // Own pawn is 1.0
    board[opp_y_transformed][opp_x_transformed] = -1.0; // Opponent pawn is -1.0

    // Flatten the 2D board into a 1D vector, Add the board values row by row
    for row in board {
        values.extend(row);
    }
    Ok(())
}

/// Encodes wall positions from the perspective of the given player
fn encode_wall_positions(game: &Game, values: &mut Vec<f64>) -> Result<()> {
    let wall_grid_size = game.board_size as usize - 1;

    // Initialize all wall positions to 0.0 (no wall)
    let mut wall_grid = vec![vec![0.0; wall_grid_size]; wall_grid_size];

    // Mark the walls
    for wall in &game.walls {
        let x = wall.position.x as usize;
        let y = wall.position.y as usize;

        if x >= wall_grid_size || y >= wall_grid_size {
            return Err(GameAdapterError::WallPositionOutOfBounds(x, y).into());
        }

        // Transform coordinates based on player perspective (same center transformation must be used)
        // 0.0->7.7 | 1.0->6.7 | 2.0->5.7 | 3.0->4.7 | 4.0->3.7 | 5.0->2.7 | 6.0->1.7 | 7.0->0.7 | 8.0
        // 0.1->7.6 | 1.1->6.6 | 2.1->5.6 | 3.1->4.6 | 4.1->3.6 | 5.1->2.6 | 6.1->1.6 | 7.1->0.6 | 8.1
        // 0.2->7.5 | 1.2->6.5 | 2.2->5.5 | 3.2->4.5 | 4.2->3.5 | 5.2->2.5 | 6.2->1.5 | 7.2->0.5 | 8.2
        // 0.3->7.4 | 1.3->6.4 | 2.3->5.4 | 3.3->4.4 | 4.3->3.4 | 5.3->2.4 | 6.3->1.4 | 7.3->0.4 | 8.3
        // 0.4->7.3 | 1.4->6.3 | 2.4->5.3 | 3.4->4.3 | 4.4->3.3 | 5.4->2.3 | 6.4->1.3 | 7.4->0.3 | 8.4
        // 0.5->7.2 | 1.5->6.2 | 2.5->5.2 | 3.5->4.2 | 4.5->3.2 | 5.5->2.2 | 6.5->1.2 | 7.5->0.2 | 8.5
        // 0.6->7.1 | 1.6->6.1 | 2.6->5.1 | 3.6->4.1 | 4.6->3.1 | 5.6->2.1 | 6.6->1.1 | 7.6->0.1 | 8.6
        // 0.7->7.0 | 1.7->6.0 | 2.7->5.0 | 3.7->4.0 | 4.7->3.0 | 5.7->2.0 | 6.7->1.0 | 7.7->0.0 | 8.7
        // 0.8      | 1.8      | 2.8      | 3.8      | 4.8      | 5.8      | 6.8      | 7.8      | 8.8
        let (x_transformed, y_transformed) =
            transform_coordinates(x, y, game.current_pawn, wall_grid_size);

        wall_grid[y_transformed][x_transformed] = match wall.orientation {
            Orientation::Horizontal => 1.0, // Horizontal wall is 1.0
            Orientation::Vertical => -1.0,  // Vertical wall is -1.0
        };
    }

    // Flatten the 2D wall grid into a 1D vector, row by row
    for row in wall_grid {
        values.extend(row);
    }
    Ok(())
}

/// Encodes the number of walls each player has left
fn encode_wall_counts(game: &Game, values: &mut Vec<f64>) {
    let own_walls = game.current_pawn().number_of_available_walls as f64;
    let opponent_walls = game.other_pawn().number_of_available_walls as f64;

    values.push(own_walls);
    values.push(opponent_walls);
}

/// Transforms coordinates to match the perspective of the player
/// For player 0, coordinates remain unchanged
/// For player 1, coordinates are flipped around the center of the board
fn transform_coordinates(
    x: usize,
    y: usize,
    player_perspective: usize,
    board_size: usize,
) -> (usize, usize) {
    if player_perspective == 0 {
        // Player 0 perspective: keep as is
        (x, y)
    } else {
        // Player 1 perspective: flip
        (board_size - 1 - x, board_size - 1 - y)
    }
}

/// Calculates the Manhattan distance from a pawn to its goal
///
/// This is useful for reward calculations based on pawn positions
pub fn distance_to_goal(game: &Game, player_index: usize) -> usize {
    let pawn = &game.pawns[player_index];
    (pawn.position.y - pawn.goal_line).abs() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use quoridor::pawn::Pawn;
    use quoridor::vector::Vector;
    use quoridor::wall::{Orientation, Wall};

    #[test]
    fn test_encode_empty_board() {
        // Create a 9x9 board with 10 walls per player
        let game = Game::new(9, 10);

        let encoded = encode_board(&game).unwrap();

        // Check dimensions
        assert_eq!(encoded.rows, 147);
        assert_eq!(encoded.columns, 1);

        // Count non-zero values (should be 2 pawns + 2 wall counts)
        let non_zero_count = encoded.values.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(non_zero_count, 4);

        // Check wall counts (should be 10 each)
        assert_eq!(encoded.values[145], 10.0);
        assert_eq!(encoded.values[146], 10.0);
    }

    #[test]
    fn test_encode_with_pawns_and_walls() {
        let board_size = 9;
        let walls_per_player = 10;

        // Create custom pawns at specific positions
        let pawns = [
            Pawn {
                position: Vector::new(4, 7),
                number_of_available_walls: walls_per_player,
                goal_line: 0,
            },
            Pawn {
                position: Vector::new(4, 1),
                number_of_available_walls: 8, // Used 2 walls
                goal_line: 8,
            },
        ];

        // Create some walls
        let walls = vec![
            Wall::new(Vector::new(3, 3), Orientation::Horizontal),
            Wall::new(Vector::new(4, 5), Orientation::Vertical),
        ];

        // Create game with custom state
        let mut game = Game {
            pawns,
            current_pawn: 0,
            board_size,
            walls,
        };

        // Encode from perspective of player 0
        let encoded = encode_board(&game).unwrap();

        // Calculate indices
        let player0_pawn_index = 7 * board_size as usize + 4; // row 7, col 4
        let player1_pawn_index = 1 * board_size as usize + 4; // row 1, col 4

        let wall_grid_size = board_size as usize - 1;
        let walls_start_index = board_size as usize * board_size as usize;

        let horizontal_wall_index = walls_start_index + 3 * wall_grid_size + 3; // row 3, col 3
        let vertical_wall_index = walls_start_index + 5 * wall_grid_size + 4; // row 5, col 4

        // Check pawn positions
        assert_eq!(encoded.values[player0_pawn_index], 1.0); // Own pawn
        assert_eq!(encoded.values[player1_pawn_index], -1.0); // Opponent pawn

        // Check wall positions
        assert_eq!(encoded.values[horizontal_wall_index], 1.0); // Horizontal wall
        assert_eq!(encoded.values[vertical_wall_index], -1.0); // Vertical wall

        // Check wall counts
        assert_eq!(encoded.values[145], 10.0); // Player 0 has 10 walls
        assert_eq!(encoded.values[146], 8.0); // Player 1 has 8 walls

        // Switch perspective to player 1
        game.current_pawn = 1;
        let encoded_p1 = encode_board(&game).unwrap();

        // board form player 1 perspective is different
        let player0_pawn_index_t =
            (board_size as usize - 1 - 7) * board_size as usize + (board_size as usize - 1 - 4); // row 1, col 4
        let player1_pawn_index_t =
            (board_size as usize - 1 - 1) * board_size as usize + (board_size as usize - 1 - 4); // row 7, col 4
        assert_eq!(encoded_p1.values[player0_pawn_index_t], -1.0); // Now opponent
        assert_eq!(encoded_p1.values[player1_pawn_index_t], 1.0); // Now self

        // Wall counts also reversed
        assert_eq!(encoded_p1.values[145], 8.0); // Player 1's own walls (8)
        assert_eq!(encoded_p1.values[146], 10.0); // Player 0's walls (10)
    }

    #[test]
    fn test_distance_to_goal() {
        let game = Game::new(9, 10);

        // Player 0 starts at position (4, 8) with goal at y=0
        // Player 1 starts at position (4, 0) with goal at y=8

        let dist_p0 = distance_to_goal(&game, 0);
        let dist_p1 = distance_to_goal(&game, 1);

        assert_eq!(dist_p0, 8); // Player 0 is 8 steps from goal
        assert_eq!(dist_p1, 8); // Player 1 is 8 steps from goal

        // Create a game with pawns at different positions
        let custom_pawns = [
            Pawn {
                position: Vector::new(4, 2), // Closer to goal
                number_of_available_walls: 10,
                goal_line: 0,
            },
            Pawn {
                position: Vector::new(4, 5), // Closer to goal
                number_of_available_walls: 10,
                goal_line: 8,
            },
        ];

        let custom_game = Game {
            pawns: custom_pawns,
            current_pawn: 0,
            board_size: 9,
            walls: vec![],
        };

        let dist_p0_custom = distance_to_goal(&custom_game, 0);
        let dist_p1_custom = distance_to_goal(&custom_game, 1);

        assert_eq!(dist_p0_custom, 2); // Player 0 is 2 steps from goal
        assert_eq!(dist_p1_custom, 3); // Player 1 is 3 steps from goal
    }
}

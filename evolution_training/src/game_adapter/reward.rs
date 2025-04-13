use quoridor::game_state::Game;
use crate::settings::Settings;

#[derive(Debug, Clone, Copy)]
pub enum RewardFunction {
    Simple,
    Symmetric,
}

/// Calculates the manhattan distance from the given player to their goal line
pub fn distance_to_goal(game: &Game, player_index: usize) -> f64 {
    let pawn = &game.pawns[player_index];
    (pawn.position.y - pawn.goal_line).abs() as f64
}

/// Calculates the reward for a player based on game state, number of moves, and settings
pub fn reward_simple_per_game(game: &Game, player_index: usize, number_of_moves_played: usize, settings: &Settings) -> f64 {
    let opponent_index = 1 -player_index;
    
    let own_distance = distance_to_goal(game, player_index);
    let opponent_distance = distance_to_goal(game, opponent_index);
    
    // Check if this player won the game = reached the goal line
    let won = own_distance == 0.0;

    // Initialize reward accumulator
    let mut reward = 0.0;
    
    // Apply win reward if player won
    if won {
        reward += settings.win_reward;
        reward += settings.per_saved_turn_reward * (settings.max_moves_per_player - number_of_moves_played) as f64;
    }
    
    // Apply distance-based rewards/penalties
    // - Penalize for own distance from goal (negative coefficient means farther = worse)
    // - Reward for opponent's distance from goal (positive coefficient means farther = better)
    reward += settings.own_distance_punishment * own_distance;
    reward += settings.other_pawn_distance_reward * opponent_distance;
    
    reward
}

/// Calculates a symmetric reward where one player's gain equals the other's loss
pub fn reward_symmetric_per_game(game: &Game, player_index: usize, number_of_moves_played: usize, settings: &Settings) -> f64 {
    let opponent_index = 1 - player_index;
    
    let own_distance = distance_to_goal(game, player_index);
    let opponent_distance = distance_to_goal(game, opponent_index);
    
    // Check if this player won or the opponent won
    let won = own_distance == 0.0;
    let lost = opponent_distance == 0.0;

    // Initialize reward accumulator
    let mut reward = 0.0;
    
    // Win/loss outcomes (zero-sum)
    if won {
        reward += settings.win_reward;
        reward += settings.per_saved_turn_reward * (settings.max_moves_per_player - number_of_moves_played) as f64;
    } else if lost {
        reward -= settings.win_reward;
        reward -= settings.per_saved_turn_reward * (settings.max_moves_per_player - number_of_moves_played) as f64;
    }
    
    // Distance-based relative advantage (progress difference)
    let relative_disadvantage = own_distance - opponent_distance;
    reward += settings.own_distance_punishment * relative_disadvantage;
    
    reward
}

#[cfg(test)]
mod tests {
    use super::*;
    use quoridor::pawn::Pawn;
    use quoridor::side::Side;
    use quoridor::vector::Vector;

    fn create_test_game() -> Game {
        Game {
            pawns: [
                Pawn::new(9, Side::Bottom, 10),
                Pawn::new(9, Side::Top, 10),
            ],
            current_pawn: 0,
            board_size: 9,
            walls: Vec::new(),
        }
    }

    fn create_test_settings() -> Settings {
        Settings::default()
            .with_reward_coefficients(100.0, -5.0, 2.0, 0.5)
            .with_max_moves_per_player(50)
    }

    #[test]
    fn test_distance_to_goal() {
        let mut game = create_test_game();
        
        // Bottom player starts at y=8, goal is y=0, so distance is 8
        assert_eq!(distance_to_goal(&game, 0), 8.0);
        
        // Move bottom player closer to goal
        game.pawns[0].position = Vector::new(4, 3);
        assert_eq!(distance_to_goal(&game, 0), 3.0);
        
        // Top player starts at y=0, goal is y=8, so distance is 8
        assert_eq!(distance_to_goal(&game, 1), 8.0);
    }

    #[test]
    fn test_reward_for_win() {
        let mut game = create_test_game();
        let settings = create_test_settings();
        
        // Move bottom player to goal (win)
        game.pawns[0].position = Vector::new(4, 0);
        
        let reward = reward_simple_per_game(&game, 0, 10, &settings);
        
        // Win reward + distance punishments/rewards (own distance is 0)
        // 100 + (0.5 * [50-40]) + (-5.0 * 0.0) + (2.0 * 8.0) = 100 + 20 + 0 + 16 = 136
        assert_eq!(reward, 136.0);
    }

    #[test]
    fn test_reward_for_loss() {
        let game = create_test_game();
        let settings = create_test_settings();
        
        let reward = reward_simple_per_game(&game, 0, 10, &settings);
        
        // No win + distance calculations
        // 0 + (-5.0 * 8.0) + (2.0 * 8.0) = - 40 + 16 = -24
        assert_eq!(reward, -24.0);
    }

    #[test]
    fn test_symmetric_reward_sum_is_zero() {
        let mut game = create_test_game();
        let settings = create_test_settings();
        
        // Set the positions to arbitrary values
        game.pawns[0].position = Vector::new(4, 2);
        game.pawns[1].position = Vector::new(5, 6);
        
        let reward_player0 = reward_symmetric_per_game(&game, 0, 15, &settings);
        let reward_player1 = reward_symmetric_per_game(&game, 1, 15, &settings);
        
        // Sum of rewards should be zero (within floating point precision)
        assert!((reward_player0 + reward_player1).abs() < 1e-10);
    }

    #[test]
    fn test_symmetric_reward_win_loss() {
        let mut game = create_test_game();
        let settings = create_test_settings();
        
        // Player 0 wins
        game.pawns[0].position = Vector::new(4, 0);
        game.pawns[1].position = Vector::new(5, 6);
        
        let reward_winner = reward_symmetric_per_game(&game, 0, 20, &settings);
        let reward_loser = reward_symmetric_per_game(&game, 1, 20, &settings);
        
        // Sum of rewards should be zero (within floating point precision)
        assert!((reward_winner + reward_loser).abs() < 1e-10);
        
        // Winner reward:
        // 100 + (0.5 * [50-20]) + (-5 * [0 - 2])   = 100 + 15 + 10 = 130
        assert_eq!(reward_winner, 125.0 );
    }
}
use anyhow::Result;

use super::error::MoveError;
use super::wall::Orientation;
use super::{pawn::Pawn, wall::Wall};
use crate::utils::{directions::Direction, side::Side, vectors::Vector, vectors::VectorUtility};
use crate::NUMBER_OF_PLAYERS;

pub enum Move {
    PawnMove(Vector),
    WallMove(Wall),
}

#[derive(Clone)]
pub struct Game {
    pub pawns: [Pawn; NUMBER_OF_PLAYERS],
    pub current_pawn: usize,
    pub board_size: i16,
    pub walls: Vec<Wall>,
}

impl Game {
    pub fn new(board_size: i16, number_of_walls_per_player: i16) -> Self {
        //initialization of the two pawns
        let pawns: [Pawn; NUMBER_OF_PLAYERS] = [
            Pawn::new(board_size, Side::Bottom, number_of_walls_per_player),
            Pawn::new(board_size, Side::Top, number_of_walls_per_player),
        ];
        Self {
            pawns,
            current_pawn: 0,
            board_size,
            walls: Vec::new(),
        }
    }

    fn current_pawn(&self) -> &Pawn {
        &self.pawns[self.current_pawn]
    }
    fn current_pawn_mut(&mut self) -> &mut Pawn {
        &mut self.pawns[self.current_pawn]
    }
    fn other_pawn(&self) -> &Pawn {
        &self.pawns[self.current_pawn + 1 % NUMBER_OF_PLAYERS]
    }
    fn set_current_pawn_next(&mut self) {
        self.current_pawn = self.current_pawn + 1 % NUMBER_OF_PLAYERS;
    }

    pub fn make_move(&mut self, player_move: Move) -> Result<()> {
        match player_move {
            Move::PawnMove(new_pos) => self.move_current_pawn(new_pos),
            Move::WallMove(new_wall) => self.place_wall(new_wall),
        }
    }

    fn move_current_pawn(&mut self, new_position: Vector) -> Result<()> {
        let allowed_positions = self.valid_next_positions();
        if !allowed_positions.contains(&new_position) {
            return Err(MoveError::InvalidPawnMove(new_position.x, new_position.y).into());
        }
        self.pawns[self.current_pawn].set_pawn_position(new_position);
        self.set_current_pawn_next();
        Ok(())
    }

    pub fn valid_next_positions(&self) -> Vec<Vector> {
        let mut res: Vec<Vector> = Vec::new();
        let current_pawn_position = self.current_pawn().position;
        for direction in [
            Direction::Up,
            Direction::Right,
            Direction::Down,
            Direction::Left,
        ] {
            self.handle_step(&mut res, direction, current_pawn_position);
        }
        res
    }

    fn handle_step(
        &self,
        result: &mut Vec<Vector>,
        move_direction: Direction,
        pawn_position: Vector,
    ) {
        let movement = move_direction.to_vector();
        let new_pos = pawn_position.add(movement);
        if !self.is_position_on_pawn_grid(new_pos) {
            return;
        };
        if self.does_wall_block_move(&move_direction, pawn_position) {
            return;
        }
        //check if move ends on other pawn
        if new_pos == self.other_pawn().position {
            self.handle_jump(result, move_direction, new_pos);
            return;
        }
        result.push(new_pos);
    }

    fn handle_jump(
        &self,
        result: &mut Vec<Vector>,
        move_direction: Direction,
        pawn_position: Vector,
    ) {
        let result_len = result.len();
        self.handle_step(result, move_direction, pawn_position);
        //if pawn cant jump forward try left and right
        if result.len() == result_len {
            self.handle_step(result, move_direction.turn_left(), pawn_position);
            self.handle_step(result, move_direction.turn_right(), pawn_position);
        }
    }

    fn is_position_on_pawn_grid(&self, position: Vector) -> bool {
        let tile_grid_size = self.board_size;
        let (x, y) = (position.x, position.y);
        x >= 0 && y >= 0 && x < tile_grid_size && y < tile_grid_size
    }

    fn is_position_on_wall_grid(&self, position: Vector) -> bool {
        let wall_grid_size = self.board_size - 1;
        let (x, y) = (position.x, position.y);
        x >= 0 && y >= 0 && x < wall_grid_size && y < wall_grid_size
    }

    pub fn is_wall_valid(&self, new_wall: &Wall) -> bool {
        let new_wall_pos = new_wall.position;
        if !self.is_position_on_wall_grid(new_wall_pos) {
            return false;
        };
        for wall in self.walls.iter() {
            if new_wall.is_in_conflict_with(wall) {
                return false;
            }
        }
        //create copy of Game, add the Wall there
        //and check if all Player can reach there goal
        let new_wall = new_wall.clone();
        let mut temp_game = self.clone();
        temp_game.walls.push(new_wall);
        if !temp_game.check_pawn_paths() {
            return false;
        }
        true
    }

    pub fn place_wall(&mut self, new_wall: Wall) -> Result<()> {
        if !self.is_wall_valid(&new_wall) {
            return Err(
                MoveError::InvalidWallMove(new_wall.position.x, new_wall.position.y).into(),
            );
        }
        if self.current_pawn().number_of_available_walls <= 0 {
            return Err(MoveError::NoWallsLeft(self.current_pawn as i16).into());
        }
        self.current_pawn_mut().dec_number_of_walls();
        self.walls.push(new_wall.clone());
        self.set_current_pawn_next();
        Ok(())
    }

    //checks if all pawns can reach the according goal line
    pub fn check_pawn_paths(&self) -> bool {
        for pawn_index in 0..NUMBER_OF_PLAYERS {
            if !self.check_pawn_path(pawn_index) {
                return false;
            }
        }
        true
    }

    //checks if pawn with given index can reach his goal line
    pub fn check_pawn_path(&self, pawn_index: usize) -> bool {
        let mut visited_positions = Vec::new();
        let goal_line = self.pawns[pawn_index].goal_line;
        visited_positions.push(self.pawns[pawn_index].position);

        let mut index: usize = 0;
        while let Some(current_position) = visited_positions.get(index) {
            //this is only possible because only two players are supported
            if goal_line == current_position.y {
                return true;
            }

            let valid_positions = self.valid_next_positions_without_other_pawn(*current_position);

            //add valid_positions to visited_positions if they are not already added
            for position in valid_positions {
                if !visited_positions.contains(&position) {
                    visited_positions.push(position);
                }
            }
            index += 1;
        }
        false
    }

    fn valid_next_positions_without_other_pawn(&self, pawn_pos: Vector) -> Vec<Vector> {
        let directions = [
            Direction::Up,
            Direction::Right,
            Direction::Down,
            Direction::Left,
        ];

        directions
            .into_iter()
            .filter_map(|dir| self.handle_step_without_other(dir, pawn_pos))
            .collect()
    }

    //same as handle_step but ignoring the other player
    fn handle_step_without_other(
        &self,
        move_direction: Direction,
        pawn_position: Vector,
    ) -> Option<Vector> {
        let movement = move_direction.to_vector();
        let new_pos = pawn_position.add(movement);
        if !self.is_position_on_pawn_grid(new_pos) {
            return None;
        };
        if self.does_wall_block_move(&move_direction, pawn_position) {
            return None;
        }
        Some(new_pos)
    }

    fn does_wall_block_move(&self, move_direction: &Direction, pawn_pos: Vector) -> bool {
        //generate the list of walls which would block the move
        let blocking_walls: Vec<Wall> = match move_direction {
            Direction::Up => [Vector::new(-1, -1), Vector::new(0, -1)]
                .into_iter()
                .map(|v| pawn_pos.add(v))
                .filter(|pos| self.is_position_on_wall_grid(*pos))
                .map(|v| Wall::new(v, Orientation::Horizontal))
                .collect(),
            Direction::Right => [Vector::new(0, -1), Vector::new(0, 0)]
                .into_iter()
                .map(|v| pawn_pos.add(v))
                .filter(|pos| self.is_position_on_wall_grid(*pos))
                .map(|v| Wall::new(v, Orientation::Vertical))
                .collect(),
            Direction::Down => [Vector::new(-1, 0), Vector::new(0, 0)]
                .into_iter()
                .map(|v| pawn_pos.add(v))
                .filter(|pos| self.is_position_on_wall_grid(*pos))
                .map(|v| Wall::new(v, Orientation::Horizontal))
                .collect(),
            Direction::Left => [Vector::new(-1, -1), Vector::new(-1, 0)]
                .into_iter()
                .map(|v| pawn_pos.add(v))
                .filter(|pos| self.is_position_on_wall_grid(*pos))
                .map(|v| Wall::new(v, Orientation::Vertical))
                .collect(),
        };
        //check if one of these walls exists
        for wall in blocking_walls.iter() {
            if self.walls.contains(&wall) {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::game_logic::wall::{Orientation, Wall};
    use crate::utils::directions::Direction;

    #[test]
    fn does_wall_block_move_positive_up_1() {
        let board_size = 9;
        let number_of_available_walls = 10;
        let pawns = [
            Pawn::new(board_size, Side::Top, number_of_available_walls),
            Pawn::new(board_size, Side::Bottom, number_of_available_walls),
        ];
        let walls = vec![Wall::new(Vector::new(3, 3), Orientation::Horizontal)];
        let game = Game {
            pawns,
            walls,
            board_size,
            current_pawn: 0,
        };
        assert!(game.does_wall_block_move(&Direction::Up, Vector::new(4, 4)));
    }

    #[test]
    fn does_wall_block_move_positive_up_2() {
        let board_size = 9;
        let number_of_available_walls = 10;
        let pawns = [
            Pawn::new(board_size, Side::Top, number_of_available_walls),
            Pawn::new(board_size, Side::Bottom, number_of_available_walls),
        ];
        let walls = vec![Wall::new(Vector::new(4, 3), Orientation::Horizontal)];
        let game = Game {
            pawns,
            walls,
            board_size,
            current_pawn: 0,
        };
        assert!(game.does_wall_block_move(&Direction::Up, Vector::new(4, 4)));
    }

    #[test]
    fn does_wall_block_move_positive_right_1() {
        let board_size = 9;
        let number_of_available_walls = 10;
        let pawns = [
            Pawn::new(board_size, Side::Top, number_of_available_walls),
            Pawn::new(board_size, Side::Bottom, number_of_available_walls),
        ];
        let walls = vec![Wall::new(Vector::new(4, 4), Orientation::Vertical)];
        let game = Game {
            pawns,
            walls,
            board_size,
            current_pawn: 0,
        };
        assert!(game.does_wall_block_move(&Direction::Right, Vector::new(4, 4)));
    }

    #[test]
    fn does_wall_block_move_positive_right_2() {
        let board_size = 9;
        let number_of_available_walls = 10;
        let pawns = [
            Pawn::new(board_size, Side::Top, number_of_available_walls),
            Pawn::new(board_size, Side::Bottom, number_of_available_walls),
        ];
        let walls = vec![Wall::new(Vector::new(4, 3), Orientation::Vertical)];
        let game = Game {
            pawns,
            walls,
            board_size,
            current_pawn: 0,
        };
        assert!(game.does_wall_block_move(&Direction::Right, Vector::new(4, 4)));
    }

    #[test]
    fn does_wall_block_move_positive_down_1() {
        let board_size = 9;
        let number_of_available_walls = 10;
        let pawns = [
            Pawn::new(board_size, Side::Top, number_of_available_walls),
            Pawn::new(board_size, Side::Bottom, number_of_available_walls),
        ];
        let walls = vec![Wall::new(Vector::new(4, 4), Orientation::Horizontal)];
        let game = Game {
            pawns,
            walls,
            board_size,
            current_pawn: 0,
        };
        assert!(game.does_wall_block_move(&Direction::Down, Vector::new(4, 4)));
    }

    #[test]
    fn does_wall_block_move_positive_down_2() {
        let board_size = 9;
        let number_of_available_walls = 10;
        let pawns = [
            Pawn::new(board_size, Side::Top, number_of_available_walls),
            Pawn::new(board_size, Side::Bottom, number_of_available_walls),
        ];
        let walls = vec![Wall::new(Vector::new(3, 4), Orientation::Horizontal)];
        let game = Game {
            pawns,
            walls,
            board_size,
            current_pawn: 0,
        };
        assert!(game.does_wall_block_move(&Direction::Down, Vector::new(4, 4)));
    }

    #[test]
    fn does_wall_block_move_positive_left_1() {
        let board_size = 9;
        let number_of_available_walls = 10;
        let pawns = [
            Pawn::new(board_size, Side::Top, number_of_available_walls),
            Pawn::new(board_size, Side::Bottom, number_of_available_walls),
        ];
        let walls = vec![Wall::new(Vector::new(3, 3), Orientation::Vertical)];
        let game = Game {
            pawns,
            walls,
            board_size,
            current_pawn: 0,
        };
        assert!(game.does_wall_block_move(&Direction::Left, Vector::new(4, 4)));
    }

    #[test]
    fn does_wall_block_move_positive_left_2() {
        let board_size = 9;
        let number_of_available_walls = 10;
        let pawns = [
            Pawn::new(board_size, Side::Top, number_of_available_walls),
            Pawn::new(board_size, Side::Bottom, number_of_available_walls),
        ];
        let walls = vec![Wall::new(Vector::new(3, 4), Orientation::Vertical)];
        let game = Game {
            pawns,
            walls,
            board_size,
            current_pawn: 0,
        };
        assert!(game.does_wall_block_move(&Direction::Left, Vector::new(4, 4)));
    }

    #[test]
    fn handle_jump_no_jump() {
        let board_size = 9;
        let number_of_available_walls = 10;
        let pawns = [
            Pawn::new(board_size, Side::Top, number_of_available_walls),
            Pawn::new(board_size, Side::Bottom, number_of_available_walls),
        ];
        let walls = vec![];
        let game = Game {
            pawns,
            walls,
            board_size,
            current_pawn: 0,
        };
        let mut act_moves = vec![];
        game.handle_jump(&mut act_moves, Direction::Up, Vector::new(4, 4));
        let exp_move_1 = Vector::new(4, 3);
        assert!(act_moves.len() == 1);
        assert!(act_moves.contains(&exp_move_1));
    }

    #[test]
    fn handle_jump_blocked_by_border() {
        let board_size = 9;
        let number_of_available_walls = 10;
        let pawns = [
            Pawn::new(board_size, Side::Top, number_of_available_walls),
            Pawn::new(board_size, Side::Bottom, number_of_available_walls),
        ];
        let walls = vec![];
        let game = Game {
            pawns,
            walls,
            board_size,
            current_pawn: 0,
        };
        let mut act_moves = vec![];
        game.handle_jump(&mut act_moves, Direction::Up, Vector::new(4, 0));
        let exp_move_1 = Vector::new(3, 0);
        let exp_move_2 = Vector::new(5, 0);
        assert!(act_moves.len() == 2);
        assert!(act_moves.contains(&exp_move_1));
        assert!(act_moves.contains(&exp_move_2));
    }

    #[test]
    fn handle_jump_corner() {
        let board_size = 9;
        let number_of_available_walls = 10;
        let pawns = [
            Pawn::new(board_size, Side::Top, number_of_available_walls),
            Pawn::new(board_size, Side::Bottom, number_of_available_walls),
        ];
        let walls = vec![];
        let game = Game {
            pawns,
            walls,
            board_size,
            current_pawn: 0,
        };
        let mut act_moves = vec![];
        game.handle_jump(&mut act_moves, Direction::Up, Vector::new(0, 0));
        let exp_move_1 = Vector::new(1, 0);
        assert!(act_moves.len() == 1);
        assert!(act_moves.contains(&exp_move_1));
    }

    #[test]
    fn handle_step_empty() {
        let board_size = 9;
        let number_of_available_walls = 10;
        let pawns = [
            Pawn::new(board_size, Side::Top, number_of_available_walls),
            Pawn::new(board_size, Side::Bottom, number_of_available_walls),
        ];
        let walls = vec![];
        let game = Game {
            pawns,
            walls,
            board_size,
            current_pawn: 0,
        };
        let mut act_moves = vec![];
        game.handle_step(&mut act_moves, Direction::Up, Vector::new(4, 4));
        let exp_move_1 = Vector::new(4, 3);
        assert!(act_moves.len() == 1);
        assert!(act_moves.contains(&exp_move_1));
    }

    #[test]
    fn handle_step_with_jump() {
        let board_size = 9;
        let number_of_available_walls = 10;
        let pawns = [
            Pawn::new(board_size, Side::Bottom, number_of_available_walls),
            Pawn {
                position: Vector::new(4, 3),
                number_of_available_walls,
                goal_line: 8,
            },
        ];
        let walls = vec![];
        let game = Game {
            pawns,
            walls,
            board_size,
            current_pawn: 0,
        };
        let mut act_moves = vec![];
        game.handle_step(&mut act_moves, Direction::Up, Vector::new(4, 4));
        let exp_move_1 = Vector::new(4, 2);
        assert!(act_moves.len() == 1);
        assert!(act_moves.contains(&exp_move_1));
    }

    #[test]
    fn handle_step_with_wall() {
        let board_size = 9;
        let number_of_available_walls = 10;
        let pawns = [
            Pawn::new(board_size, Side::Top, number_of_available_walls),
            Pawn::new(board_size, Side::Bottom, number_of_available_walls),
        ];
        let walls = vec![Wall::new(Vector::new(3, 3), Orientation::Horizontal)];
        let game = Game {
            pawns,
            walls,
            board_size,
            current_pawn: 0,
        };

        let mut act_moves = vec![];
        game.handle_step(&mut act_moves, Direction::Up, Vector::new(4, 4));
        assert!(act_moves.len() == 0);
    }

    #[test]
    fn handle_step_without_other_empty() {
        let board_size = 9;
        let number_of_available_walls = 10;
        let pawns = [
            Pawn::new(board_size, Side::Top, number_of_available_walls),
            Pawn::new(board_size, Side::Bottom, number_of_available_walls),
        ];
        let walls = vec![];
        let game = Game {
            pawns,
            walls,
            board_size,
            current_pawn: 0,
        };
        let act_move: Option<Vector> =
            game.handle_step_without_other(Direction::Up, Vector::new(4, 4));
        let exp_move = Vector::new(4, 3);
        assert_eq!(act_move, Some(exp_move));
    }

    #[test]
    fn handle_step_without_other_with_pawn() {
        let board_size = 9;
        let number_of_available_walls = 10;
        let pawns = [
            Pawn::new(board_size, Side::Bottom, number_of_available_walls),
            Pawn {
                position: Vector::new(4, 3),
                number_of_available_walls,
                goal_line: 8,
            },
        ];
        let walls = vec![];
        let game = Game {
            pawns,
            walls,
            board_size,
            current_pawn: 0,
        };
        let act_move: Option<Vector> =
            game.handle_step_without_other(Direction::Up, Vector::new(4, 4));
        let exp_move = Vector::new(4, 3);
        assert_eq!(act_move, Some(exp_move));
    }

    #[test]
    fn handle_step_without_other_with_wall() {
        let board_size = 9;
        let number_of_available_walls = 10;
        let pawns = [
            Pawn::new(board_size, Side::Top, number_of_available_walls),
            Pawn::new(board_size, Side::Bottom, number_of_available_walls),
        ];
        let walls = vec![Wall::new(Vector::new(3, 3), Orientation::Horizontal)];
        let game = Game {
            pawns,
            walls,
            board_size,
            current_pawn: 0,
        };
        let act_move: Option<Vector> =
            game.handle_step_without_other(Direction::Up, Vector::new(4, 4));
        assert_eq!(act_move, None);
    }

    #[test]
    fn valid_next_positions_without_other_pawn_empty() {
        let board_size = 9;
        let number_of_available_walls = 10;
        let pawns = [
            Pawn::new(board_size, Side::Top, number_of_available_walls),
            Pawn::new(board_size, Side::Bottom, number_of_available_walls),
        ];
        let walls = vec![];
        let game = Game {
            pawns,
            walls,
            board_size,
            current_pawn: 0,
        };
        let act_moves = game.valid_next_positions_without_other_pawn(Vector::new(4, 4));
        let exp_moves = [
            Vector::new(4, 3),
            Vector::new(4, 5),
            Vector::new(3, 4),
            Vector::new(5, 4),
        ];
        assert!(act_moves.len() == 4);
        assert!(act_moves.contains(&exp_moves[0]));
        assert!(act_moves.contains(&exp_moves[1]));
        assert!(act_moves.contains(&exp_moves[2]));
        assert!(act_moves.contains(&exp_moves[3]));
    }
}

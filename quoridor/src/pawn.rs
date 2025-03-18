use crate::side::Side;
use crate::vector::{Vector, VectorUtility};

#[derive(Clone)]
pub struct Pawn {
    pub position: Vector,
    pub goal_line: i16,
    pub number_of_available_walls: i16,
}

impl Pawn {
    pub fn new(board_size: i16, pawn_side: Side, number_of_available_walls: i16) -> Self {
        let position: Vector = Self::calculate_start_coordinate(board_size, &pawn_side);
        let goal_line: i16 = Self::calculate_goal_line(board_size, &pawn_side);
        Self {
            position,
            goal_line,
            number_of_available_walls,
        }
    }

    fn calculate_start_coordinate(board_size: i16, pawn_side: &Side) -> Vector {
        //half_board is the index at the half of the board | Example board_size = 9 => 0 1 2 3 4 5 6 7 8 => half_board = 4
        //board_size needs to be uneven for this to work
        let half_board: i16 = board_size / 2;
        match pawn_side {
            Side::Top => Vector::new(half_board, 0),
            Side::Bottom => Vector::new(half_board, board_size - 1),
        }
    }

    fn calculate_goal_line(board_size: i16, pawn_side: &Side) -> i16 {
        match pawn_side {
            Side::Top => board_size - 1,
            Side::Bottom => 0,
        }
    }

    pub fn set_pawn_position(&mut self, new_position: Vector) {
        self.position = new_position;
    }
    pub fn move_pawn(&mut self, movement: Vector) {
        self.position = self.position.add(movement);
    }
    pub fn inc_number_of_walls(&mut self) {
        self.number_of_available_walls = self.number_of_available_walls + 1;
    }
    pub fn dec_number_of_walls(&mut self) {
        self.number_of_available_walls = self.number_of_available_walls - 1;
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::side::Side::*;

    #[test]
    fn calculate_start_coordinate_bottom() {
        let act_start_coordinate = Pawn::calculate_start_coordinate(9, &Bottom);
        assert_eq!(act_start_coordinate, Vector::new(4, 8));
    }

    #[test]
    fn calculate_start_coordinate_top() {
        let act_start_coordinate = Pawn::calculate_start_coordinate(9, &Top);
        assert_eq!(act_start_coordinate, Vector::new(4, 0));
    }

    #[test]
    fn calculate_goal_line_bottom() {
        let act_goal_line = Pawn::calculate_goal_line(9, &Bottom);
        assert_eq!(act_goal_line, 0);
    }

    #[test]
    fn calculate_goal_line_top() {
        let act_goal_line = Pawn::calculate_goal_line(9, &Top);
        assert_eq!(act_goal_line, 8);
    }

    #[test]
    fn new_bottom() {
        let board_size = 9;
        let pawn_side = Bottom;
        let number_of_walls = 10;
        let act_pawn = Pawn::new(board_size, pawn_side, number_of_walls);
        assert_eq!(act_pawn.position, Vector::new(4, 8));
        assert_eq!(act_pawn.goal_line, 0);
        assert_eq!(act_pawn.number_of_available_walls, 10);
    }

    #[test]
    fn new_top() {
        let board_size = 9;
        let pawn_side = Top;
        let number_of_walls = 10;
        let act_pawn = Pawn::new(board_size, pawn_side, number_of_walls);
        assert_eq!(act_pawn.position, Vector::new(4, 0));
        assert_eq!(act_pawn.goal_line, 8);
        assert_eq!(act_pawn.number_of_available_walls, 10);
    }

    #[test]
    fn new_bottom_small() {
        let board_size = 5;
        let pawn_side = Bottom;
        let number_of_walls = 7;
        let act_pawn = Pawn::new(board_size, pawn_side, number_of_walls);
        assert_eq!(act_pawn.position, Vector::new(2, 4));
        assert_eq!(act_pawn.goal_line, 0);
        assert_eq!(act_pawn.number_of_available_walls, 7);
    }

    #[test]
    fn new_top_small() {
        let board_size = 5;
        let pawn_side = Top;
        let number_of_walls = 7;
        let act_pawn = Pawn::new(board_size, pawn_side, number_of_walls);
        assert_eq!(act_pawn.position, Vector::new(2, 0));
        assert_eq!(act_pawn.goal_line, 4);
        assert_eq!(act_pawn.number_of_available_walls, 7);
    }

    #[test]
    fn move_pawn_1() {
        let mut act_pawn = Pawn {
            position: Vector::new(3, 1),
            number_of_available_walls: 0,
            goal_line: 0,
        };
        act_pawn.move_pawn(Vector::new(2, -1));
        assert_eq!(act_pawn.position, Vector::new(5, 0));
    }

    #[test]
    fn move_pawn_2() {
        let mut act_pawn = Pawn {
            position: Vector::new(0, 1),
            number_of_available_walls: 0,
            goal_line: 0,
        };
        act_pawn.move_pawn(Vector::new(2, 0));
        assert_eq!(act_pawn.position, Vector::new(2, 1));
    }

    #[test]
    fn inc_number_of_walls() {
        let mut act_pawn = Pawn {
            position: Vector::new(0, 0),
            number_of_available_walls: 2,
            goal_line: 0,
        };
        act_pawn.inc_number_of_walls();
        assert_eq!(act_pawn.number_of_available_walls, 3);
    }

    #[test]
    fn dec_number_of_walls() {
        let mut act_pawn = Pawn {
            position: Vector::new(0, 0),
            number_of_available_walls: 2,
            goal_line: 0,
        };
        act_pawn.dec_number_of_walls();
        assert_eq!(act_pawn.number_of_available_walls, 1);
    }
}

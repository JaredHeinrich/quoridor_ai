use crate::utils::vectors::{Vector, VectorUtility};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Vertical,
    Horizontal,
}

#[derive(Clone, PartialEq, Eq)]
pub struct Wall {
    pub position: Vector,
    pub orientation: Orientation,
}

impl Wall {
    //constructor
    pub fn new(position: Vector, orientation: Orientation) -> Self {
        Self {
            position,
            orientation,
        }
    }

    //returns the directional vector of the wall
    fn directional_vector(&self) -> Vector {
        match self.orientation {
            Orientation::Vertical => Vector::new(0, 1),
            Orientation::Horizontal => Vector::new(1, 0),
        }
    }

    //checks if a wall is in conflict with another wall
    pub fn is_in_conflict_with(&self, wall: &Wall) -> bool {
        let is_parallel = self.orientation == wall.orientation;
        let pos_s = self.position;
        let pos_w = wall.position;
        let dv_s = self.directional_vector();
        //Walls must not intersect or overlap
        if pos_s == pos_w {
            return true;
        };
        if !is_parallel {
            return false;
        }
        if pos_s.add(dv_s) == pos_w || pos_s.subtract(dv_s) == pos_w {
            return true;
        };
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn is_in_conflict_with_positive_same_position_same_direction() {
        let wall_1 = Wall::new(Vector::new(5, 5), Orientation::Horizontal);
        let wall_2 = Wall::new(Vector::new(5, 5), Orientation::Horizontal);
        assert!(wall_1.is_in_conflict_with(&wall_2));
    }

    #[test]
    fn is_in_conflict_with_positive_same_position_different_direction() {
        let wall_1 = Wall::new(Vector::new(5, 5), Orientation::Horizontal);
        let wall_2 = Wall::new(Vector::new(5, 5), Orientation::Vertical);
        assert!(wall_1.is_in_conflict_with(&wall_2));
    }

    #[test]
    fn is_in_conflict_with_positive_different_position_same_direction_horizontal_1() {
        let wall_1 = Wall::new(Vector::new(5, 5), Orientation::Horizontal);
        let wall_2 = Wall::new(Vector::new(6, 5), Orientation::Horizontal);
        assert!(wall_1.is_in_conflict_with(&wall_2));
    }

    #[test]
    fn is_in_conflict_with_positive_different_position_same_direction_horizontal_2() {
        let wall_1 = Wall::new(Vector::new(5, 5), Orientation::Horizontal);
        let wall_2 = Wall::new(Vector::new(4, 5), Orientation::Horizontal);
        assert!(wall_1.is_in_conflict_with(&wall_2));
    }

    #[test]
    fn is_in_conflict_with_positive_different_position_same_direction_vertical_1() {
        let wall_1 = Wall::new(Vector::new(5, 5), Orientation::Vertical);
        let wall_2 = Wall::new(Vector::new(5, 6), Orientation::Vertical);
        assert!(wall_1.is_in_conflict_with(&wall_2));
    }

    #[test]
    fn is_in_conflict_with_positive_different_position_same_direction_vertical_2() {
        let wall_1 = Wall::new(Vector::new(5, 5), Orientation::Vertical);
        let wall_2 = Wall::new(Vector::new(5, 4), Orientation::Vertical);
        assert!(wall_1.is_in_conflict_with(&wall_2));
    }

    #[test]
    fn is_in_conflict_with_negative_different_position_same_direction_1() {
        let wall_1 = Wall::new(Vector::new(5, 5), Orientation::Vertical);
        let wall_2 = Wall::new(Vector::new(5, 3), Orientation::Vertical);
        assert!(!wall_1.is_in_conflict_with(&wall_2));
    }

    #[test]
    fn is_in_conflict_with_negative_different_position_same_direction_2() {
        let wall_1 = Wall::new(Vector::new(5, 5), Orientation::Horizontal);
        let wall_2 = Wall::new(Vector::new(7, 5), Orientation::Horizontal);
        assert!(!wall_1.is_in_conflict_with(&wall_2));
    }

    #[test]
    fn is_in_conflict_with_negative_different_position_different_direction_1() {
        let wall_1 = Wall::new(Vector::new(5, 5), Orientation::Horizontal);
        let wall_2 = Wall::new(Vector::new(6, 5), Orientation::Vertical);
        assert!(!wall_1.is_in_conflict_with(&wall_2));
    }

    #[test]
    fn is_in_conflict_with_negative_different_position_different_direction_2() {
        let wall_1 = Wall::new(Vector::new(5, 5), Orientation::Horizontal);
        let wall_2 = Wall::new(Vector::new(5, 6), Orientation::Vertical);
        assert!(!wall_1.is_in_conflict_with(&wall_2));
    }
}

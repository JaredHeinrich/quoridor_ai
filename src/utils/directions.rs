use crate::utils::vectors::Vector;

//Eine Richtung in welche sich z.B. der Spieler bewegen kann.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Direction {
    Up,
    Right,
    Down,
    Left,
}

impl Direction {
    //Die Richtung in einen "Einheitsvector" umwandeln.
    pub fn to_vector(&self) -> Vector {
        match self {
            Self::Up => Vector::new(0, -1),
            Self::Right => Vector::new(1, 0),
            Self::Down => Vector::new(0, 1),
            Self::Left => Vector::new(-1, 0),
        }
    }

    //Die Richtung umkehren
    pub fn revert(&self) -> Self {
        match self {
            Self::Up => Self::Down,
            Self::Down => Self::Up,
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }

    //Die Richtung nach links drehen
    pub fn turn_left(&self) -> Self {
        match self {
            Self::Up => Self::Left,
            Self::Right => Self::Up,
            Self::Down => Self::Right,
            Self::Left => Self::Down,
        }
    }

    //Die Richtung nach rechts drehen
    pub fn turn_right(&self) -> Self {
        match self {
            Self::Up => Self::Right,
            Self::Right => Self::Down,
            Self::Down => Self::Left,
            Self::Left => Self::Up,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_vector_up() {
        let direction = Direction::Up;
        let act_vector = direction.to_vector();
        let exp_vector = Vector::new(0, -1);
        assert_eq!(act_vector, exp_vector);
    }

    #[test]
    fn to_vector_right() {
        let direction = Direction::Right;
        let act_vector = direction.to_vector();
        let exp_vector = Vector::new(1, 0);
        assert_eq!(act_vector, exp_vector);
    }

    #[test]
    fn to_vector_down() {
        let direction = Direction::Down;
        let act_vector = direction.to_vector();
        let exp_vector = Vector::new(0, 1);
        assert_eq!(act_vector, exp_vector);
    }

    #[test]
    fn to_vector_left() {
        let direction = Direction::Left;
        let act_vector = direction.to_vector();
        let exp_vector = Vector::new(-1, 0);
        assert_eq!(act_vector, exp_vector);
    }

    #[test]
    fn revert_up() {
        let direction = Direction::Up;
        let act_direction = direction.revert();
        let exp_direction = Direction::Down;
        assert_eq!(act_direction, exp_direction);
    }

    #[test]
    fn revert_right() {
        let direction = Direction::Right;
        let act_direction = direction.revert();
        let exp_direction = Direction::Left;
        assert_eq!(act_direction, exp_direction);
    }

    #[test]
    fn revert_down() {
        let direction = Direction::Down;
        let act_direction = direction.revert();
        let exp_direction = Direction::Up;
        assert_eq!(act_direction, exp_direction);
    }

    #[test]
    fn revert_left() {
        let direction = Direction::Left;
        let act_direction = direction.revert();
        let exp_direction = Direction::Right;
        assert_eq!(act_direction, exp_direction);
    }

    #[test]
    fn turn_left_up() {
        let direction = Direction::Up;
        let act_direction = direction.turn_left();
        let exp_direction = Direction::Left;
        assert_eq!(act_direction, exp_direction);
    }

    #[test]
    fn turn_left_right() {
        let direction = Direction::Right;
        let act_direction = direction.turn_left();
        let exp_direction = Direction::Up;
        assert_eq!(act_direction, exp_direction);
    }

    #[test]
    fn turn_left_down() {
        let direction = Direction::Down;
        let act_direction = direction.turn_left();
        let exp_direction = Direction::Right;
        assert_eq!(act_direction, exp_direction);
    }

    #[test]
    fn turn_left_left() {
        let direction = Direction::Left;
        let act_direction = direction.turn_left();
        let exp_direction = Direction::Down;
        assert_eq!(act_direction, exp_direction);
    }

    #[test]
    fn turn_right_up() {
        let direction = Direction::Up;
        let act_direction = direction.turn_right();
        let exp_direction = Direction::Right;
        assert_eq!(act_direction, exp_direction);
    }

    #[test]
    fn turn_right_right() {
        let direction = Direction::Right;
        let act_direction = direction.turn_right();
        let exp_direction = Direction::Down;
        assert_eq!(act_direction, exp_direction);
    }

    #[test]
    fn turn_right_down() {
        let direction = Direction::Down;
        let act_direction = direction.turn_right();
        let exp_direction = Direction::Left;
        assert_eq!(act_direction, exp_direction);
    }

    #[test]
    fn turn_right_left() {
        let direction = Direction::Left;
        let act_direction = direction.turn_right();
        let exp_direction = Direction::Up;
        assert_eq!(act_direction, exp_direction);
    }
}

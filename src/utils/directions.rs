use crate::utils::vectors::Vector;

//Eine Richtung in welche sich z.B. der Spieler bewegen kann.
#[derive(Debug, PartialEq)]
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

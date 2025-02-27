//Darstellung eines Vektors.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Vector {
    x: i16,
    y: i16,
}

pub trait VectorUtility {
    fn add(&self, vector: Self) -> Self;
    fn subtract(&self, vector: Self) -> Self;
    fn revert(&self) -> Self;
}

impl Vector {
    //Konstruktor für den Vektor
    pub fn new(x: i16, y: i16) -> Self {
        Vector { x, y }
    }

    //Getter
    pub fn x(&self) -> i16 {
        self.x
    }

    pub fn y(&self) -> i16 {
        self.y
    }
    //Getter
}

//implementation der Vector Utility
impl VectorUtility for Vector {
    //Addiert zwei Vektoren und gibt das Ergebniss zurück.
    fn add(&self, vector: Self) -> Self {
        Vector {
            x: self.x + vector.x,
            y: self.y + vector.y,
        }
    }
    //Subtrahiert zwei Vektoren und gibt das Ergebniss zurück.
    fn subtract(&self, vector: Self) -> Self {
        Vector {
            x: self.x - vector.x,
            y: self.y - vector.y,
        }
    }
    //Multipliziert einen Vektor mit -1
    fn revert(&self) -> Self {
        Vector {
            x: -self.x,
            y: -self.y,
        }
    }
}

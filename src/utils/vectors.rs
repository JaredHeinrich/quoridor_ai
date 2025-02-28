//Darstellung eines Vektors.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Vector {
    pub x: i16,
    pub y: i16,
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

#[cfg(test)]
impl Default for Vector {
    fn default() -> Self {
        Self {
            x: i16::default(),
            y: i16::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_1() {
        let v1 = Vector::new(0, 0);
        let v2 = Vector::new(0, 0);
        let exp_vector = Vector::new(0, 0);
        let act_vector = v1.add(v2);
        assert_eq!(exp_vector, act_vector);
    }

    #[test]
    fn add_2() {
        let v1 = Vector::new(10, 3);
        let v2 = Vector::new(-5, 2);
        let exp_vector = Vector::new(5, 5);
        let act_vector = v1.add(v2);
        assert_eq!(exp_vector, act_vector);
    }

    #[test]
    fn subtract_1() {
        let v1 = Vector::new(0, 0);
        let v2 = Vector::new(0, 0);
        let exp_vector = Vector::new(0, 0);
        let act_vector = v1.subtract(v2);
        assert_eq!(exp_vector, act_vector);
    }

    #[test]
    fn subtract_2() {
        let v1 = Vector::new(12, 6);
        let v2 = Vector::new(6, -6);
        let exp_vector = Vector::new(6, 12);
        let act_vector = v1.subtract(v2);
        assert_eq!(exp_vector, act_vector);
    }

    #[test]
    fn revert_1() {
        let vector = Vector::new(0, 0);
        let exp_vector = Vector::new(0, 0);
        let act_vector = vector.revert();
        assert_eq!(exp_vector, act_vector);
    }

    #[test]
    fn revert_2() {
        let vector = Vector::new(1, -1);
        let exp_vector = Vector::new(-1, 1);
        let act_vector = vector.revert();
        assert_eq!(exp_vector, act_vector);
    }
}

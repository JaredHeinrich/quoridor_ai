use std::f64::consts::E;

pub fn sigmoid(x: f64) -> f64 {
    let epowx = E.powf(x);
    return epowx / (epowx + 1.0);
}
pub fn relu(x: f64) -> f64 {
    if 0.0 >= x {
        return 0.0;
    }
    return x;
}

#[cfg(test)]
mod tests {
    use crate::activation::{relu, sigmoid};

    #[test]
    fn test_relu_1() {
        let x: f64 = 1.0;
        assert_eq!(relu(x), x);
    }

    #[test]
    fn test_relu_2() {
        let x: f64 = 23.4;
        assert_eq!(relu(x), x);
    }

    #[test]
    fn test_relu_3() {
        let x: f64 = 0.0;
        assert_eq!(relu(x), 0.0);
    }

    #[test]
    fn test_relu_4() {
        let x: f64 = -1.0;
        assert_eq!(relu(x), 0.0);
    }

    #[test]
    fn test_relu_5() {
        let x: f64 = -12.2345;
        assert_eq!(relu(x), 0.0);
    }

    #[test]
    fn test_sigmoid_1() {
        let x: f64 = 0.0;
        assert_eq!(sigmoid(x), 0.5);
    }

    #[test]
    fn test_sigmoid_2() {
        let x: f64 = 0.5;
        assert_eq!(sigmoid(x), 0.6224593312018546);
    }

    #[test]
    fn test_sigmoid_3() {
        let x: f64 = 1.5;
        assert_eq!(sigmoid(x), 0.8175744761936437);
    }

    #[test]
    fn test_sigmoid_4() {
        let x: f64 = -0.5;
        assert_eq!(sigmoid(x), 0.37754066879814546);
    }

    #[test]
    fn test_sigmoid_5() {
        let x: f64 = -1.5;
        assert_eq!(sigmoid(x), 0.18242552380635635);
    }
}

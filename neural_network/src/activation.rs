use std::f64::consts::E;

use matrix::matrix::Matrix;

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

/// Converts neural network outputs to probabilities using softmax
pub fn softmax_maxtrick(x_vec: Matrix) -> Matrix {
    // Find maximum value for numerical stability (e.g. e^200 explodes)
    let max_val = x_vec.values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    // Calculate e^(x_i - max_val) for each element
    let exp_values: Vec<f64> = x_vec.values.iter()
        .map(|&val| E.powf(val - max_val))
        .collect();
    
    // Calculate the sum of all exp values
    let sum_exp: f64 = exp_values.iter().sum();
    
    // Normalize by dividing each exp value by the sum
    let normalized_values: Vec<f64> = exp_values.iter()
        .map(|&val| val / sum_exp)
        .collect();
    
    // Create and return the result matrix with the same dimensions as the input
    Matrix::new(x_vec.rows, x_vec.columns, normalized_values).unwrap()
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

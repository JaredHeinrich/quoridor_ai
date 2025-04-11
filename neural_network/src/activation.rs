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
    use crate::activation::{relu, sigmoid, softmax_maxtrick};
    use matrix::matrix::Matrix;

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

    #[test]
    fn test_softmax_probabilities() {
        let input = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
        let output = softmax_maxtrick(input);
        
        // All values should be between 0 and 1
        for val in &output.values {
            assert!(0.0 <= *val && *val <= 1.0);
        }
        
        // Sum should be close to 1.0 (allowing for floating point precision)
        let sum: f64 = output.values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        
        // Higher input values should have higher probabilities
        assert!(output.values[0] < output.values[1]);
        assert!(output.values[1] < output.values[2]);
    }

    #[test]
    fn test_softmax_shape() {
        let input = Matrix::new(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output = softmax_maxtrick(input);
        
        // Output shape should match input shape
        assert_eq!(output.rows, 6);
        assert_eq!(output.columns, 1);
    }

    #[test]
    fn test_softmax_dominance() {
        // With very large differences, the largest value should dominate
        let input = Matrix::new(1, 3, vec![1.0, 10.0, 100.0]).unwrap();
        let output = softmax_maxtrick(input);
        
        // The last value should be close to 1.0
        assert!(output.values[2] > 0.999);
        
        // Other values should be close to 0
        assert!(output.values[0] < 0.001);
        assert!(output.values[1] < 0.001);
    }

    #[test]
    fn test_softmax_equal_values() {
        // For equal inputs, outputs should be uniform
        let input = Matrix::new(1, 4, vec![2.0, 2.0, 2.0, 2.0]).unwrap();
        let output = softmax_maxtrick(input);
        
        // All values should be approximately 0.25
        for val in &output.values {
            assert!((*val - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Test with very large values that would normally cause overflow
        let input = Matrix::new(1, 3, vec![1000.0, 1000.0, 1000.0]).unwrap();
        let output = softmax_maxtrick(input);
        
        // All values should be approximately 1/3
        for val in &output.values {
            assert!((*val - (1.0/3.0)).abs() < 1e-10);
        }
    }
}

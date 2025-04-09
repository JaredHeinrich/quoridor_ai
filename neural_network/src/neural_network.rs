use std::{f64, process::Output};

use anyhow::Result;
use matrix::matrix::Matrix;

use crate::{
    activation::{relu, sigmoid},
    error::NNError,
};

#[derive(Debug)]
pub struct NeuralNetwork {
    pub layer_sizes: Vec<usize>,
    pub weights: Vec<Matrix>,
    pub biases: Vec<Matrix>,
}

impl NeuralNetwork {
    /// creates a neural network from given layer sizes.
    /// layer 0: input layer
    /// layer 1..n-1: hidden layers
    /// layer n: output layer
    ///
    /// # Example:
    /// ```
    /// use neural_network::neural_network::NeuralNetwork;
    /// let nn = NeuralNetwork::new(&vec![3, 1, 1]);
    /// ```
    /// The created network will take an input vector with 3 values.
    /// Has 1 hidden layer with 1 node, and 1 output layer with 1 node.
    ///
    /// The biases for each layer are stored in a list of n x 1 matrices, where n is the number of
    /// nodes in the specific layer.
    ///
    /// The weights for all connections between two layers are stored in list of n x m matrices,
    /// where n is the number of nodes in the specific layer and m is the number of nodes in the
    /// previous layer.
    ///
    /// weights[0]  biases[0]  weights[1]  biases[1]
    ///   x x x        x           x          x
    pub fn new(layer_sizes: &Vec<usize>) -> Result<Self> {
        if layer_sizes.len() < 2 {
            return Err(NNError::CreationTooFewLayersError(layer_sizes.len()).into());
        }
        if let Some(index) = layer_sizes.iter().find(|size| **size == 0) {
            return Err(NNError::CreationEmptyLayerError(*index).into());
        }
        let layer_sizes = layer_sizes.clone();

        let (weights, biases): (Vec<Matrix>, Vec<Matrix>) = layer_sizes
            .windows(2)
            .map(|window| {
                let prev_size: usize = window[0];
                let current_size: usize = window[1];
                (
                    Matrix::random(current_size, prev_size), // weights
                    Matrix::zero(current_size, 1),           // biases
                )
            })
            .unzip();
        Ok(NeuralNetwork {
            layer_sizes,
            weights,
            biases,
        })
    }

    /// Calculates the output vector for specific input vector.
    /// For the hidden layers the next output vector is calculated with:
    /// ReLU(weights * input_vector + biases)
    ///
    /// For the output layer sigmoid is used as activation function instead:
    /// Sigmoid(weights * input_vector + biases)
    pub fn feed_forward(&self, input_vector: Matrix) -> Result<Matrix> {
        if input_vector.columns != 1 {
            return Err(
                NNError::InvalidInputVectorShape(input_vector.rows, input_vector.columns).into(),
            );
        }
        if input_vector.rows != self.layer_sizes[0] {
            return Err(
                NNError::InvalidInputVectorSize(self.layer_sizes[1], input_vector.rows).into(),
            );
        }

        let mut result = input_vector;
        let mut layer_iter = self.weights.iter().zip(self.biases.iter());
        let (output_layer_weigth_matrix, output_layer_bias_matrix) =
            layer_iter.next_back().unwrap();

        while let Some((weight_matrix, bias_matrix)) = layer_iter.next() {
            process_layer(&mut result, weight_matrix, bias_matrix, relu);
        }
        process_layer(
            &mut result,
            output_layer_weigth_matrix,
            output_layer_bias_matrix,
            sigmoid,
        );
        Ok(result)
    }

    /// Mutate all the weights and biases of the neural network.
    pub fn mutate(&mut self, mutation_rate: f64) {
        for weight_matrix in &mut self.weights {
            weight_matrix.mutate_all(mutation_rate);
        }
        for bias_matrix in &mut self.biases {
            bias_matrix.mutate_all(mutation_rate);
        }
    }
}

fn process_layer(
    result_buffer: &mut Matrix,
    weight_matrix: &Matrix,
    bias_matrix: &Matrix,
    activation_function: impl Fn(f64) -> f64,
) {
    *result_buffer = weight_matrix
        .multiply(&result_buffer)
        .unwrap()
        .add(bias_matrix)
        .unwrap();
    result_buffer
        .values
        .iter_mut()
        .for_each(|value| *value = activation_function(*value));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_zero_layers() {
        let layer_sizes = vec![];
        let nn = NeuralNetwork::new(&layer_sizes);
        assert!(nn.is_err());
        let error = nn.unwrap_err();
        let error = error.downcast::<NNError>();
        assert!(matches!(error, Ok(NNError::CreationTooFewLayersError(_))));
        if let Ok(NNError::CreationTooFewLayersError(x)) = error {
            assert_eq!(x, layer_sizes.len());
        }
    }

    #[test]
    fn test_new_one_layer() {
        let layer_sizes = vec![1];
        let nn = NeuralNetwork::new(&layer_sizes);
        assert!(nn.is_err());
        let error = nn.unwrap_err();
        let error = error.downcast::<NNError>();
        assert!(matches!(error, Ok(NNError::CreationTooFewLayersError(_))));
        if let Ok(NNError::CreationTooFewLayersError(x)) = error {
            assert_eq!(x, layer_sizes.len());
        }
    }

    #[test]
    fn test_new_empty_layer() {
        let layer_sizes = vec![1, 2, 0];
        let nn = NeuralNetwork::new(&layer_sizes);
        assert!(nn.is_err());
        let error = nn.unwrap_err();
        let error = error.downcast::<NNError>();
        assert!(matches!(error, Ok(NNError::CreationEmptyLayerError(_))));
        if let Ok(NNError::CreationTooFewLayersError(x)) = error {
            assert_eq!(x, *layer_sizes.iter().find(|v| **v == 0).unwrap());
        }
    }

    #[test]
    fn test_new() {
        let layer_sizes = vec![3, 2, 2, 2];
        let nn = NeuralNetwork::new(&layer_sizes);
        assert!(nn.is_ok());
        let nn = nn.unwrap();
        assert_eq!(nn.layer_sizes, layer_sizes);
        assert_eq!(nn.weights.len(), layer_sizes.len() - 1);
        assert_eq!(nn.biases.len(), layer_sizes.len() - 1);
        assert!(nn.weights.iter().all(|bias_matrix| bias_matrix
            .values
            .iter()
            .all(|value| *value >= -1.0 && *value <= 1.0)));
        assert!(nn
            .biases
            .iter()
            .all(|bias_matrix| bias_matrix.values.iter().all(|value| *value == 0.0)));
    }

    #[test]
    fn test_mutate_zero_rate() {
        let layer_sizes = vec![2, 3, 1];
        let mut nn = NeuralNetwork::new(&layer_sizes).unwrap();
        
        // Clone original weights and biases
        let original_weights: Vec<Vec<f64>> = nn.weights.iter().map(|m| m.values.clone()).collect();
        let original_biases: Vec<Vec<f64>> = nn.biases.iter().map(|m| m.values.clone()).collect();
        
        nn.mutate(0.0);
        
        // Verify that weights and biases remain unchanged
        for (i, weight_matrix) in nn.weights.iter().enumerate() {
            assert_eq!(weight_matrix.values, original_weights[i]);
        }
        for (i, bias_matrix) in nn.biases.iter().enumerate() {
            assert_eq!(bias_matrix.values, original_biases[i]);
        }
    }

    #[test]
    fn test_mutate_changes_values() {
        let layer_sizes = vec![2, 3, 1];
        let mut nn = NeuralNetwork::new(&layer_sizes).unwrap();
        
        // Clone the original weights and biases
        let original_weights: Vec<Vec<f64>> = nn.weights.iter().map(|m| m.values.clone()).collect();
        let original_biases: Vec<Vec<f64>> = nn.biases.iter().map(|m| m.values.clone()).collect();

        // Apply mutation with significant rate
        nn.mutate(1.0);
        
        // Verify that at least some weights have changed
        let mut any_weight_changed = false;
        for (i, weight_matrix) in nn.weights.iter().enumerate() {
            if weight_matrix.values != original_weights[i] {
                any_weight_changed = true;
                break;
            }
        }
        assert!(any_weight_changed, "No weights changed after mutation");

        // Verify that at least some biases have changed
        let mut any_bias_changed = false;
        for (i, bias_matrix) in nn.biases.iter().enumerate() {
            if bias_matrix.values != original_biases[i] {
                any_bias_changed = true;
                break;
            }
        }
        assert!(any_bias_changed, "No biases changed after mutation");
    }

    #[test]
    fn test_mutate_respects_rate() {
        // Create a neural network
        let layer_sizes = vec![2, 3, 1];
        let mut nn = NeuralNetwork::new(&layer_sizes).unwrap();
        
        // Initialize all weights and biases to zero
        for weight_matrix in &mut nn.weights {
            for value in &mut weight_matrix.values {
                *value = 0.0;
            }
        }
        for bias_matrix in &mut nn.biases {
            for value in &mut bias_matrix.values {
                *value = 0.0;
            }
        }
        
        let mutation_rate = 0.5;
        nn.mutate(mutation_rate);
        
        // Verify that all mutations are within bounds
        for weight_matrix in &nn.weights {
            for &value in &weight_matrix.values {
                assert!(value.abs() <= mutation_rate, "Mutation exceeded rate bounds: {}", value);
            }
        }
        for bias_matrix in &nn.biases {
            for &value in &bias_matrix.values {
                assert!(value.abs() <= mutation_rate, "Mutation exceeded rate bounds: {}", value);
            }
        }
    }
}

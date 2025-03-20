use anyhow::Result;
use matrix::matrix::Matrix;

use crate::error::NNError;

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
    /// let nn = NeuralNetwork::new(vec![3, 1, 1]);
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
    pub fn new(layer_sizes: Vec<usize>) -> Result<Self> {
        if layer_sizes.len() < 2 {
            return Err(NNError::CreationToFewLayersError(layer_sizes.len()).into());
        }
        if let Some(index) = layer_sizes.iter().find(|size| **size == 0) {
            return Err(NNError::CreationEmptyLayerError(*index).into());
        }

        let (weights, biases): (Vec<Matrix>, Vec<Matrix>) = layer_sizes
            .windows(2)
            .map(|window| {
                let prev_size: usize = window[0];
                let current_size: usize = window[1];
                (
                    Matrix::random(current_size, prev_size), // weights
                    Matrix::zero(current_size, 1), // biases
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
            return Err(NNError::InvalidInputVectorSize(self.layer_sizes[1], input_vector.rows).into());
        }

        let mut result = input_vector;
        self.weights
            .iter()
            .zip(self.biases.iter())
            .for_each(|(weight_matrix, bias_matrix)| {
                result = weight_matrix
                    .multiply(&result)
                    .unwrap()
                    .add(bias_matrix)
                    .unwrap();
            });
        todo!("Use Sigmoid or ReLU function");
        Ok(result)
    }
}

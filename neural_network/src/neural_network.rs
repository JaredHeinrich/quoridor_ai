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
    ///
    /// 311 will look something like this:
    ///
    ///   0 0 1 1: indices in weights and biases list
    /// 0 \
    /// 0 - 1 - 2
    /// 0 /
    ///
    /// Layer 0 is the input layer and wont be stored in the neural network.
    /// But it is needed to create the weight matrix between layer 0 and layer 1.
    /// In this scenario the matrices look like the following.
    ///
    /// weights[0]  biases[0]  weights[1]  biases[1]
    ///
    ///   x x x        x           x          x
    ///
    /// The goal is that weights[0] * input_layer + biases[0] will result in the output values from
    /// layer 1
    ///
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
    /// For each Layer the next output vector is calculated this way:
    /// weights * input_vector + biases
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
        Ok(result)
    }
}

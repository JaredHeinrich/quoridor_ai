use anyhow::Result;
use matrix::matrix::Matrix;

use crate::error::NNError;

pub struct NeuralNetwork {
    pub layers: Vec<usize>,
    pub weights: Vec<Matrix>,
    pub biases: Vec<Matrix>,
}

impl NeuralNetwork {
    /// 0      : input layer
    /// 1..n-1 : hidden layers
    /// n      : output layer
    ///
    /// creates a neural network from given layer sizes.
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
    pub fn new(layers: Vec<usize>) -> Result<Self> {
        if layers.len() < 2 {
            return Err(NNError::CreationToFewLayersError(layers.len()).into());
        }
        if let Some(index) = layers.iter().find(|layer| **layer == 0) {
            return Err(NNError::CreationEmptyLayerError(*index).into());
        }

        let (weights, biases): (Vec<Matrix>, Vec<Matrix>) = layers
            .windows(2)
            .map(|window| {
                let prev_nodes: usize = window[0];
                let current_nodes: usize = window[1];
                (
                    Matrix::random(current_nodes, prev_nodes),
                    Matrix::zero(current_nodes, 1),
                )
            })
            .unzip();
        Ok(NeuralNetwork {
            layers,
            weights,
            biases,
        })
    }

    /// Calculates the output values for specific input values.
    /// For each Layer the next output values are calculated this way:
    /// weights * input_values + biases
    pub fn process(&self, input_values: Matrix) -> Result<Matrix> {
        if input_values.columns != 1 {
            return Err(
                NNError::InvalidInputVectorShape(input_values.rows, input_values.columns).into(),
            );
        }
        if input_values.rows != self.layers[1] {
            return Err(NNError::InvalidInputVectorSize(self.layers[1], input_values.rows).into());
        }

        let mut result = input_values;
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

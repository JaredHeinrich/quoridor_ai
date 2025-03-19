use matrix::matrix::Matrix;

pub struct NeuralNetwork {
    pub layers: Vec<usize>,
    pub weights: Vec<Matrix>,
    pub biases: Vec<Matrix>,
}

impl NeuralNetwork {
    /// 0      : input layer
    /// 1..n-1 : hidden layers
    /// n      : output layer
    pub fn new(layers: Vec<usize>) -> Self {
        todo!("not implemented")
    }
}

use thiserror::Error;

#[derive(Error, Debug)]
pub enum NNError {
    #[error("A neural network needs at least 2 layers. Tried to create network with {0} layers.")]
    CreationTooFewLayersError(usize),

    #[error("A neural network can't have a layer with 0 nodes. Layer {0} in requested Network has 0 nodes.")]
    CreationEmptyLayerError(usize),

    #[error("Input vector needs to be a n x 1 matrix. Give vector is {0} x {1} matrix.")]
    InvalidInputVectorShape(usize, usize),

    #[error("Input vector doesn't match the expected size for this neural network. Expected size is {0}, actuall size is {1}.")]
    InvalidInputVectorSize(usize, usize),
}

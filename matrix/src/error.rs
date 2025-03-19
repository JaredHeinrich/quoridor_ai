use thiserror::Error;

#[derive(Error, Debug)]
pub enum MatrixError {
    #[error("Can't multiply matrices because number of columns of first matrix is not eqaual to number of rows of second matrix")]
    DotProductError,

    #[error("Can't add matrices with different dimensions")]
    AdditionError,

    #[error("Can't create matrix, because rows * columns != number of values -> {0} * {1} != {2}")]
    CreationError(usize, usize, usize),
}

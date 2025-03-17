use thiserror::Error;

#[derive(Error, Debug)]
pub enum MoveError {
    #[error("Invalid Pawn Move to [{0}, {1}]")]
    InvalidPawnMove(i16, i16),

    #[error("Invalid Wall Move to [{0}, {1}]")]
    InvalidWallMove(i16, i16),

    #[error("Player {0} has no walls left")]
    NoWallsLeft(i16),
}

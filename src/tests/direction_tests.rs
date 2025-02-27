use crate::utils::{directions::Direction, vectors::Vector};

#[test]
fn to_vector_up() {
    let direction = Direction::Up;
    let act_vector = direction.to_vector();
    let exp_vector = Vector::new(0, -1);
    assert_eq!(act_vector, exp_vector);
}

#[test]
fn to_vector_right() {
    let direction = Direction::Right;
    let act_vector = direction.to_vector();
    let exp_vector = Vector::new(1, 0);
    assert_eq!(act_vector, exp_vector);
}

#[test]
fn to_vector_down() {
    let direction = Direction::Down;
    let act_vector = direction.to_vector();
    let exp_vector = Vector::new(0, 1);
    assert_eq!(act_vector, exp_vector);
}

#[test]
fn to_vector_left() {
    let direction = Direction::Left;
    let act_vector = direction.to_vector();
    let exp_vector = Vector::new(-1, 0);
    assert_eq!(act_vector, exp_vector);
}

#[test]
fn revert_up() {
    let direction = Direction::Up;
    let act_direction = direction.revert();
    let exp_direction = Direction::Down;
    assert_eq!(act_direction, exp_direction);
}

#[test]
fn revert_right() {
    let direction = Direction::Right;
    let act_direction = direction.revert();
    let exp_direction = Direction::Left;
    assert_eq!(act_direction, exp_direction);
}

#[test]
fn revert_down() {
    let direction = Direction::Down;
    let act_direction = direction.revert();
    let exp_direction = Direction::Up;
    assert_eq!(act_direction, exp_direction);
}

#[test]
fn revert_left() {
    let direction = Direction::Left;
    let act_direction = direction.revert();
    let exp_direction = Direction::Right;
    assert_eq!(act_direction, exp_direction);
}

#[test]
fn turn_left_up() {
    let direction = Direction::Up;
    let act_direction = direction.turn_left();
    let exp_direction = Direction::Left;
    assert_eq!(act_direction, exp_direction);
}

#[test]
fn turn_left_right() {
    let direction = Direction::Right;
    let act_direction = direction.turn_left();
    let exp_direction = Direction::Up;
    assert_eq!(act_direction, exp_direction);
}

#[test]
fn turn_left_down() {
    let direction = Direction::Down;
    let act_direction = direction.turn_left();
    let exp_direction = Direction::Right;
    assert_eq!(act_direction, exp_direction);
}

#[test]
fn turn_left_left() {
    let direction = Direction::Left;
    let act_direction = direction.turn_left();
    let exp_direction = Direction::Down;
    assert_eq!(act_direction, exp_direction);
}

#[test]
fn turn_right_up() {
    let direction = Direction::Up;
    let act_direction = direction.turn_right();
    let exp_direction = Direction::Right;
    assert_eq!(act_direction, exp_direction);
}

#[test]
fn turn_right_right() {
    let direction = Direction::Right;
    let act_direction = direction.turn_right();
    let exp_direction = Direction::Down;
    assert_eq!(act_direction, exp_direction);
}

#[test]
fn turn_right_down() {
    let direction = Direction::Down;
    let act_direction = direction.turn_right();
    let exp_direction = Direction::Left;
    assert_eq!(act_direction, exp_direction);
}

#[test]
fn turn_right_left() {
    let direction = Direction::Left;
    let act_direction = direction.turn_right();
    let exp_direction = Direction::Up;
    assert_eq!(act_direction, exp_direction);
}

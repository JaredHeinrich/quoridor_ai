use crate::utils::vector::{Vector, VectorUtility};

#[test]
fn add_1() {
    let v1 = Vector::new(0, 0);
    let v2 = Vector::new(0, 0);
    let exp_vector = Vector::new(0, 0);
    let act_vector = v1.add(v2);
    assert_eq!(exp_vector, act_vector);
}

#[test]
fn add_2() {
    let v1 = Vector::new(10, 3);
    let v2 = Vector::new(-5, 2);
    let exp_vector = Vector::new(5, 5);
    let act_vector = v1.add(v2);
    assert_eq!(exp_vector, act_vector);
}

#[test]
fn subtract_1() {
    let v1 = Vector::new(0, 0);
    let v2 = Vector::new(0, 0);
    let exp_vector = Vector::new(0, 0);
    let act_vector = v1.subtract(v2);
    assert_eq!(exp_vector, act_vector);
}

#[test]
fn subtract_2() {
    let v1 = Vector::new(12, 6);
    let v2 = Vector::new(6, -6);
    let exp_vector = Vector::new(6, 12);
    let act_vector = v1.subtract(v2);
    assert_eq!(exp_vector, act_vector);
}

#[test]
fn revert_1() {
    let vector = Vector::new(0, 0);
    let exp_vector = Vector::new(0, 0);
    let act_vector = vector.revert();
    assert_eq!(exp_vector, act_vector);
}

#[test]
fn revert_2() {
    let vector = Vector::new(1, -1);
    let exp_vector = Vector::new(-1, 1);
    let act_vector = vector.revert();
    assert_eq!(exp_vector, act_vector);
}

use anyhow::Result;
use rand::Rng;

use crate::error::MatrixError;

#[derive(Debug)]
pub struct Matrix {
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, columns: usize, values: Vec<f64>) -> Result<Self> {
        if columns * rows != values.len() {
            return Err(MatrixError::CreationError(rows, columns, values.len()).into());
        }
        Ok(Self {
            rows,
            columns,
            values,
        })
    }

    /// Creates Matrix with random values from -1.0 to 1.0
    pub fn random(rows: usize, columns: usize) -> Self {
        let mut values = vec![0.0; rows * columns];
        values
            .iter_mut()
            .for_each(|value| *value = rand::rng().random_range(-1.0..1.0));
        Self {
            rows,
            columns,
            values,
        }
    }

    /// Creates Matrix with all values 0.0
    pub fn zero(rows: usize, columns: usize) -> Self {
        Self {
            rows,
            columns,
            values: vec![0.0; rows * columns],
        }
    }

    /// Creates Matrix with all values 0.0
    pub fn multiply(&self, other: &Self) -> Result<Self> {
        if self.columns != other.rows {
            return Err(MatrixError::DotProductError.into());
        }
        let mut result = Self::new(
            self.rows,
            other.columns,
            vec![0.0; self.rows * other.columns],
        )
        .unwrap();

        for row_index in 0..result.rows {
            for column_index in 0..result.columns {
                *result.value_mut(column_index, row_index) = self
                    .iter_row(row_index)
                    .zip(other.iter_column(column_index))
                    .map(|(x, y)| x * y)
                    .sum();
            }
        }
        Ok(result)
    }

    /// Adds two matrices per element
    ///
    /// Returns error if matrices have different dimensions.
    /// Can only add n x m matrix to n x m matrix.
    pub fn add(&self, other: &Self) -> Result<Self> {
        if self.rows != other.rows || self.columns != other.columns {
            return Err(MatrixError::AdditionError.into());
        }
        let values = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(x, y)| x + y)
            .collect();
        Self::new(self.rows, self.columns, values)
    }

    /// Returns refrence to value of matrix at specific row and column
    pub fn value(&self, row_index: usize, column_index: usize) -> &f64 {
        &self.values[self.calculate_index(row_index, column_index)]
    }

    /// Returns mut refrence to value of matrix at specific row and column
    pub fn value_mut(&mut self, row_index: usize, column_index: usize) -> &mut f64 {
        let index = self.calculate_index(row_index, column_index);
        &mut self.values[index]
    }

    /// Iterate over rows of matrix
    pub fn iter_rows(&self) -> impl Iterator<Item = impl Iterator<Item = &f64>> {
        self.values
            .chunks(self.columns)
            .into_iter()
            .map(|chunk| chunk.iter())
    }

    /// Iterate over columns of matrix
    pub fn iter_columns(&self) -> impl Iterator<Item = impl Iterator<Item = &f64>> {
        (0..self.columns).map(move |column_index| {
            (0..self.rows)
                .map(move |row_index| &self.values[self.calculate_index(row_index, column_index)])
        })
    }

    /// Iterate over specific row of matrix
    pub fn iter_row(&self, row_index: usize) -> impl Iterator<Item = &f64> {
        let row_start = self.calculate_index(row_index, 0);
        let row_end = self.calculate_index(row_index + 1, 0);
        (row_start..row_end).map(|index| &self.values[index])
    }

    /// Iterate over specific column of matrix
    pub fn iter_column(&self, column_index: usize) -> impl Iterator<Item = &f64> {
        (0..self.rows)
            .map(move |row_index| &self.values[self.calculate_index(row_index, column_index)])
    }

    /// Calculate index from row and column
    pub fn calculate_index(&self, row_index: usize, column_index: usize) -> usize {
        column_index + self.columns * row_index
    }

    /// Mutate all values of the matrix by adding random value from -1.0 to 1.0 times mutation_rate
    pub fn mutate_all(&mut self, mutation_rate: f64) {
        for value in &mut self.values {
            *value += rand::rng().random_range(-1.0..1.0) * mutation_rate;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_positiv() {
        let matrix = Matrix::new(2, 3, vec![-1.0, 2.0, 3.0, 4.4, -5.123, 6.0]).unwrap();
        assert_eq!(matrix.columns, 3);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.values, vec![-1.0, 2.0, 3.0, 4.4, -5.123, 6.0]);
    }

    #[test]
    fn test_new_wrong_value_length() {
        let matrix = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(matrix.is_err());
    }

    #[test]
    fn test_calculate_index() {
        //  |  0  1  2  3  4 c
        //--+---------------
        // 0|  0  1  2  3  4
        // 1|  5  6  7  8  9
        // 2| 10 11 12 13 14
        // r
        let rows = 3;
        let columns = 5;
        let matrix: Matrix = Matrix {
            rows,
            columns,
            values: vec![0.0; rows * columns],
        };
        assert_eq!(0, matrix.calculate_index(0, 0));
        assert_eq!(4, matrix.calculate_index(0, 4));
        assert_eq!(6, matrix.calculate_index(1, 1));
        assert_eq!(10, matrix.calculate_index(2, 0));
        assert_eq!(12, matrix.calculate_index(2, 2));
        assert_eq!(14, matrix.calculate_index(2, 4));
    }

    #[test]
    fn test_iter_row() {
        //  |  0  1  2  3  4 c
        //--+---------------
        // 0|  0  1  2  3  4
        // 1|  5  6  7  8  9
        // 2| 10 11 12 13 14
        // r
        let rows = 3;
        let columns = 5;
        let matrix: Matrix = Matrix {
            rows,
            columns,
            values: vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
            ],
        };
        let mut iterator = matrix.iter_row(0);
        assert_eq!(Some(&0.0), iterator.next());
        assert_eq!(Some(&1.0), iterator.next());
        assert_eq!(Some(&2.0), iterator.next());
        assert_eq!(Some(&3.0), iterator.next());
        assert_eq!(Some(&4.0), iterator.next());
        assert!(iterator.next().is_none());

        assert_eq!(60.0, matrix.iter_row(2).sum());
        assert_eq!(
            vec![&5.0, &6.0, &7.0, &8.0, &9.0],
            matrix.iter_row(1).collect::<Vec<&f64>>()
        );
    }

    #[test]
    fn test_iter_column() {
        //  |  0  1  2  3  4 c
        //--+---------------
        // 0|  0  1  2  3  4
        // 1|  5  6  7  8  9
        // 2| 10 11 12 13 14
        // r
        let rows = 3;
        let columns = 5;
        let matrix: Matrix = Matrix {
            rows,
            columns,
            values: vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
            ],
        };
        let mut iterator = matrix.iter_column(0);
        assert_eq!(Some(&0.0), iterator.next());
        assert_eq!(Some(&5.0), iterator.next());
        assert_eq!(Some(&10.0), iterator.next());
        assert!(iterator.next().is_none());

        assert_eq!(21.0, matrix.iter_column(2).sum());
        assert_eq!(
            vec![&1.0, &6.0, &11.0],
            matrix.iter_column(1).collect::<Vec<&f64>>()
        );
    }

    #[test]
    fn test_iter_rows() {
        //  |  0  1  2  3  4 c
        //--+---------------
        // 0|  0  1  2  3  4
        // 1|  5  6  7  8  9
        // 2| 10 11 12 13 14
        // r
        let rows = 3;
        let columns = 5;
        let matrix: Matrix = Matrix {
            rows,
            columns,
            values: vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
            ],
        };
        let mut iterator = matrix.iter_rows();
        assert_eq!(
            vec![&0.0, &1.0, &2.0, &3.0, &4.0],
            iterator.next().unwrap().collect::<Vec<&f64>>()
        );
        assert_eq!(
            vec![&5.0, &6.0, &7.0, &8.0, &9.0],
            iterator.next().unwrap().collect::<Vec<&f64>>()
        );
        assert_eq!(
            vec![&10.0, &11.0, &12.0, &13.0, &14.0],
            iterator.next().unwrap().collect::<Vec<&f64>>()
        );
        assert!(iterator.next().is_none());
    }

    #[test]
    fn test_iter_columns() {
        //  |  0  1  2  3  4 c
        //--+---------------
        // 0|  0  1  2  3  4
        // 1|  5  6  7  8  9
        // 2| 10 11 12 13 14
        // r
        let rows = 3;
        let columns = 5;
        let matrix: Matrix = Matrix {
            rows,
            columns,
            values: vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
            ],
        };
        let mut iterator = matrix.iter_columns();
        assert_eq!(
            vec![&0.0, &5.0, &10.0],
            iterator.next().unwrap().collect::<Vec<&f64>>()
        );
        assert_eq!(
            vec![&1.0, &6.0, &11.0],
            iterator.next().unwrap().collect::<Vec<&f64>>()
        );
        assert_eq!(
            vec![&2.0, &7.0, &12.0],
            iterator.next().unwrap().collect::<Vec<&f64>>()
        );
        assert_eq!(
            vec![&3.0, &8.0, &13.0],
            iterator.next().unwrap().collect::<Vec<&f64>>()
        );
        assert_eq!(
            vec![&4.0, &9.0, &14.0],
            iterator.next().unwrap().collect::<Vec<&f64>>()
        );
        assert!(iterator.next().is_none());
    }

    #[test]
    fn test_multiply() {
        // matrix 1
        //  |  0  1  2  3  c
        //--+-------------
        // 0|  0  1  2  3
        // 1|  4  5  6  7
        // r
        //
        // matrix 2
        //   |  0  c
        // --+----
        //  0|  8
        //  1|  9
        //  2| 10
        //  3| 11
        //  r
        //
        // expected result
        //   |  0  c
        // --+----
        //  0| 62 = 0 + 9+ 20 + 33
        //  1|214 = 32 + 45 + 60 + 77
        //  r

        let matrix_one = Matrix {
            rows: 2,
            columns: 4,
            values: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        };

        let matrix_two = Matrix {
            rows: 4,
            columns: 1,
            values: vec![8.0, 9.0, 10.0, 11.0],
        };
        let expected_result = Matrix {
            rows: 2,
            columns: 1,
            values: vec![62.0, 214.0],
        };
        let actual_result = matrix_one.multiply(&matrix_two).unwrap();
        assert_eq!(expected_result.rows, actual_result.rows);
        assert_eq!(expected_result.columns, actual_result.columns);
        assert_eq!(expected_result.values, actual_result.values);
    }

    #[test]
    fn test_multiply_negativ() {
        // matrix 1
        //  |  0  1  2  3  c
        //--+-------------
        // 0|  0  1  2  3
        // 1|  4  5  6  7
        // r
        //
        // matrix 2
        //   |  0  c
        // --+----
        //  0|  8
        //  1|  9
        //  2| 10
        //  r

        let matrix_one = Matrix {
            rows: 2,
            columns: 4,
            values: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        };

        let matrix_two = Matrix {
            rows: 3,
            columns: 1,
            values: vec![8.0, 9.0, 10.0],
        };
        let actual_result = matrix_one.multiply(&matrix_two);
        assert!(actual_result.is_err());
    }

    #[test]
    fn test_add() {
        // matrix 1
        //  |  0  1  2  c
        //--+----------
        // 0|  0  1  2
        // 1|  3  4  5
        // r
        //
        // matrix 2
        //  |  0  1  2  c
        //--+----------
        // 0|  5  4  3
        // 1|  2  1  0
        // r
        //
        // expected result
        //  |  0  1  2  c
        //--+----------
        // 0|  5  5  5
        // 1|  5  5  5
        // r

        let matrix_one = Matrix {
            rows: 2,
            columns: 3,
            values: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        };

        let matrix_two = Matrix {
            rows: 2,
            columns: 3,
            values: vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        };

        let expected_result = Matrix {
            rows: 2,
            columns: 3,
            values: vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        };
        let actual_result = matrix_one.add(&matrix_two).unwrap();

        assert_eq!(expected_result.rows, actual_result.rows);
        assert_eq!(expected_result.columns, actual_result.columns);
        assert_eq!(expected_result.values, actual_result.values);
    }

    #[test]
    fn test_add_negative() {
        // matrix 1
        //  |  0  1  2  c
        //--+----------
        // 0|  0  1  2
        // 1|  3  4  5
        // r
        //
        // matrix 2
        //  |  0  1  2  c
        //--+----------
        // 0|  2  1  0
        // r

        let matrix_one = Matrix {
            rows: 2,
            columns: 3,
            values: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        };

        let matrix_two = Matrix {
            rows: 1,
            columns: 3,
            values: vec![2.0, 1.0, 0.0],
        };

        let actual_result = matrix_one.add(&matrix_two);
        assert!(actual_result.is_err())
    }

    #[test]
    fn test_mutate_all_zero_rate() {
        let mut matrix = Matrix {
            rows: 2,
            columns: 2,
            values: vec![1.0, 2.0, 3.0, 4.0],
        };

        let original_values = matrix.values.clone();

        matrix.mutate_all(0.0);

        assert_eq!(matrix.values, original_values);
    }

    #[test]
    fn test_mutate_all_changes_values() {
       let mut matrix = Matrix {
            rows: 3,
            columns: 3,
            values: vec![0.0; 9],
        };
        
        matrix.mutate_all(0.01);
        
        // At least some values should have changed from 0.0
        assert!(matrix.values.iter().any(|&v| v != 0.0));
    }
    
    #[test]
    fn test_mutate_all_both_directions() {
        let mut matrix = Matrix {
            rows: 10,
            columns: 10,
            values: vec![0.0; 100],
        };
        
        matrix.mutate_all(1.0);
        
        // Check that values can be both positive and negative
        assert!(matrix.values.iter().any(|&v| v > 0.0));
        assert!(matrix.values.iter().any(|&v| v < 0.0));
    }

    #[test]
    fn test_mutate_all_respects_mutation_rate() {
        let mut matrix = Matrix {
            rows: 3,
            columns: 3,
            values: vec![0.0; 9],
        };
        
        let mutatation_rate = 0.5;
        matrix.mutate_all(mutatation_rate);
        
        // All values should be within the bounds of ±mutation_rate
        for value in &matrix.values {
            assert!(value.abs() <= mutatation_rate);
        }
    }
}

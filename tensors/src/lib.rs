use anyhow::Error;
use interfaces::tensors::{Element, Tensor};
use std::{fmt::Debug, ops::{Add, Mul}, vec::Vec};

#[derive(Debug, Clone)]
struct TensorImpl<E>
where
    E: Element,
{
    shape: Vec<usize>,
    data: Vec<E>,
}

/// Adding to two tensors together.
impl<E: Element> Add for TensorImpl<E> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        if self.shape() != other.shape() {
            panic!("Shapes are not compatible for element-wise operations.");
        }

        let data: Vec<E> = self
            .data
            .iter()
            .zip(other.data.iter())
            // TODO(mhauru) What's the consequence of cloning here? Does it affect performance?
            .map(|(a, b)| a.clone() + b.clone())
            .collect();
        // TODO: Remove the unwrap, and return a Result instead
        TensorImpl::from_vec(&self.shape(), &data).unwrap()
    }
}

/// Adding to a scalar to a tensors together.
impl<E: Element> Add<E> for TensorImpl<E> {
    type Output = Self;

    fn add(self, scalar: E) -> Self {
        let data: Vec<E> = self
            .data
            .iter()
            // TODO(mhauru) What's the consequence of cloning here? Does it affect performance?
            .map(|a| a.clone() + scalar.clone())
            .collect();
        // TODO: Remove the unwrap, and return a Result instead
        TensorImpl::from_vec(&self.shape(), &data).unwrap()
    }
}

/// Multiplying to two tensors together elementwise.
impl<E: Element> Mul for TensorImpl<E> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        if self.shape() != other.shape() {
            panic!("Shapes are not compatible for element-wise operations.");
        }

        let data: Vec<E> = self
            .data
            .iter()
            .zip(other.data.iter())
            // TODO(mhauru) What's the consequence of cloning here? Does it affect performance?
            .map(|(a, b)| a.clone() * b.clone())
            .collect();
        // TODO: Remove the unwrap, and return a Result instead
        TensorImpl::from_vec(&self.shape(), &data).unwrap()
    }
}

/// Multiplying tensor by a scalar.
impl<E: Element> Mul<E> for TensorImpl<E> {
    type Output = Self;

    fn mul(self, scalar: E) -> Self {
        let data: Vec<E> = self
            .data
            .iter()
            // TODO(mhauru) What's the consequence of cloning here? Does it affect performance?
            .map(|a| a.clone() * scalar.clone())
            .collect();
        // TODO: Remove the unwrap, and return a Result instead
        TensorImpl::from_vec(&self.shape(), &data).unwrap()
    }
}

impl<E> Tensor<E> for TensorImpl<E>
where
    E: Element,
{
    type TensorError = Error;

    fn from_vec(shape: &Vec<usize>, data: &Vec<E>) -> Result<Self, Self::TensorError> {
        if shape.iter().product::<usize>() != data.len() {
            return Err(Error::msg(
                "The length of the `data` param does not match the values of the `shape` param",
            ));
        } else {
            Ok(TensorImpl {
                shape:shape.clone(),
                data:data.clone()
            })
        }
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    ///// Fill a matrix by repeatedly cloning the provided element.
    ///// Note: the behaviour might be unexpected if the provided element clones "by reference".
    //fn fill_with_clone(shape: Vec<usize>, element: E) -> Self {}

    //fn at(&self, idxs: Vec<usize>) -> Option<&E>;

    //fn at_mut(&mut self, idxs: Vec<usize>) -> Option<&mut E>;

    fn transpose(&self) -> Self {
        // Swap the last two elements of the shape vector
        let mut shape = self.shape.clone();
        let num_dims = shape.len();
        shape.swap(num_dims - 1, num_dims - 2);
        // TODO(mhauru) Cloning here feel unnecessary, can we get rid of it?
        let leading_dims = shape.clone().into_iter().take(num_dims - 2).reduce(|a, b| a * b).unwrap();

        let n_elements = self.data.len();
        let mut data : Vec<E> = Vec::with_capacity(n_elements);
        for i in 0..leading_dims {
            for j in 0..shape[num_dims - 2] {
                for k in 0..shape[num_dims - 1] {
                    let transposed_idx = i * shape[num_dims - 1] * shape[num_dims - 2] + k * shape[num_dims - 2] + j;
                    data.push(self.data[transposed_idx].clone());
                }
            }
        }
        TensorImpl { shape, data }
    }

    //fn matmul(&self, other: &Self) -> Result<Self, Self::TensorError>;

    ///// Sum across one or more dimensions (eg. row-wise sum for a 2D matrix resulting in a "column
    ///// vector")
    //fn dim_sum(&self, dim: Vec<usize>) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_vec() {
        let shape = vec![2, 3];
        let data = vec![1, 2, 3, 4, 5, 6];
        let maybe_tensor = TensorImpl::from_vec(&shape, &data);
        let tensor = maybe_tensor.unwrap();

        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.data, data);
    }

    #[test]
    fn test_from_vec_invalid_params() {
        // The length of the `data` Vec is not that implied by the `shape` Vec
        let shape = vec![2, 3];
        let data = vec![1, 2];
        let maybe_tensor = TensorImpl::from_vec(&shape, &data);
        assert!(maybe_tensor.is_err());
        let err = maybe_tensor.unwrap_err();

        assert!(err.to_string().contains(
            "The length of the `data` param does not match the values of the `shape` param"
        ));
    }

    #[test]
    fn test_shape() {
        let shape = vec![2, 3];
        let data = vec![1, 2, 3, 4, 5, 6];
        let tensor = TensorImpl::from_vec(&shape, &data);

        assert_eq!(tensor.unwrap().shape(), shape);
    }

    #[test]
    fn test_transpose_1d_tensor() {
        let shape = vec![2];
        let data = vec![1, 2];
        let tensor = TensorImpl::from_vec(&shape, &data).unwrap();
        let transposed = tensor.transpose();
        assert_eq!(transposed.shape(), shape);
        // Transposing a 1D tensor should return the original tensor
        assert_eq!(transposed.data, data);
    }

    #[test]
    fn test_transpose_2d_tensor() {
        // Case 1
        let shape1 = vec![2, 3];
        let data1 = vec![1, 2, 3, 4, 5, 6];
        let tensor1 = TensorImpl::from_vec(&shape1, &data1).unwrap();
        let transposed1 = tensor1.transpose();
        assert_eq!(transposed1.shape(), vec![3, 2]);

        // Case 2
        let shape2 = vec![2, 2];
        let original_data2 = vec![1, 2, 3, 4];
        let tensor2 = TensorImpl::from_vec(&shape2, &original_data2).unwrap();
        let transposed2 = tensor2.transpose();
        assert_eq!(transposed2.shape(), vec![2, 2]);

        let expected_data2 = vec![1, 3, 2, 4];
        assert_eq!(transposed2.data, expected_data2);

    }

    #[test]
    fn test_transpose_3d_tensor() {
        let shape = vec![3, 4, 5];
        let original_data = (1..61).collect::<Vec<i32>>();

        let tensor = TensorImpl::from_vec(&shape, &original_data).unwrap();
        let transposed = tensor.transpose();
        let expected_shape = vec![3, 5, 4];
        assert_eq!(transposed.shape(), expected_shape);

        // let expected_data = vec![1, 2, 3, 4, 5, 7, 6, 8];
        // assert_eq!(transposed.data, expected_data);

        // Transposing twice should return the original tensor
        let transposed_twice = transposed.transpose();
        assert_eq!(transposed_twice.shape(), shape);
        assert_eq!(transposed_twice.data, original_data);
    }


    // Addition
    #[test]
    fn test_adding_tensors() {
        let shape = vec![2, 3];
        let data1 = vec![1, 2, 3, 4, 5, 6];
        let data2 = vec![10, 20, 30, 40, 50, 60];
        let tensor1 = TensorImpl::from_vec(&shape, &data1).unwrap();
        let tensor2 = TensorImpl::from_vec(&shape, &data2).unwrap();

        let tensor3 = tensor1 + tensor2;
        assert_eq!(tensor3.data, vec![11, 22, 33, 44, 55, 66]);
    }

    #[test]
    fn test_adding_tensors_wrong_shapes() {
        let shape1 = vec![2, 3];
        let shape2 = vec![2, 2];
        let data1 = vec![1, 2, 3, 4, 5, 6];
        let data2 = vec![10, 20, 30, 40];
        let tensor1 = TensorImpl::from_vec(&shape1, &data1).unwrap();
        let tensor2 = TensorImpl::from_vec(&shape2, &data2).unwrap();

        assert!(std::panic::catch_unwind(|| tensor1 + tensor2).is_err());
    }

    #[test]
    fn test_adding_tensors_and_scalars() {
        let shape = vec![2, 3];
        let data = vec![1, 2, 3, 4, 5, 6];
        let tensor = TensorImpl::from_vec(&shape, &data).unwrap();

        let tensor2 = tensor + 10;
        assert_eq!(tensor2.data, vec![11, 12, 13, 14, 15, 16]);
    }

    // Element-wise multiplication
    #[test]
    fn test_multiplying_tensors() {
        let shape = vec![2, 3];
        let data1 = vec![1, 2, 3, 4, 5, 6];
        let data2 = vec![10, 20, 30, 40, 50, 60];
        let tensor1 = TensorImpl::from_vec(&shape, &data1).unwrap();
        let tensor2 = TensorImpl::from_vec(&shape, &data2).unwrap();

        let tensor3 = tensor1 * tensor2;
        assert_eq!(tensor3.data, vec![10, 40, 90, 160, 250, 360]);
    }

    #[test]
    fn test_multiplying_tensors_wrong_shapes() {
        let shape1 = vec![2, 3];
        let shape2 = vec![2, 2];
        let data1 = vec![1, 2, 3, 4, 5, 6];
        let data2 = vec![10, 20, 30, 40];
        let tensor1 = TensorImpl::from_vec(&shape1, &data1).unwrap();
        let tensor2 = TensorImpl::from_vec(&shape2, &data2).unwrap();

        assert!(std::panic::catch_unwind(|| tensor1 * tensor2).is_err());
    }

    #[test]
    fn test_multiplying_tensors_and_scalars() {
        let shape = vec![2, 3];
        let data = vec![1, 2, 3, 4, 5, 6];
        let tensor = TensorImpl::from_vec(&shape, &data).unwrap();

        let tensor2 = tensor * 10;
        assert_eq!(tensor2.data, vec![10, 20, 30, 40, 50, 60]);
    }
}

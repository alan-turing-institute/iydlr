use interfaces::tensors::{Element, Tensor};
use std::{
    ops::Add,
    vec::Vec,
    fmt::Debug,
};
use anyhow::Error;

#[derive(Debug, Clone)]
struct TensorImpl<E>
where
    E: Element,
{
    shape: Vec<usize>,
    data: Vec<E>,
}

/// Adding to two tensors together.
impl<E: Element> Add for TensorImpl<E>
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        if self.shape() != other.shape() {
            panic!("Shapes are not compatible for addition");
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            // TODO(mhauru) What's the consequence of cloning here? Does it affect performance?
            .map(|(a, b)| a.clone() + b.clone())
            .collect();
        // TODO: Remove the unwrap, and return a Result instead
        TensorImpl::from_vec(self.shape(), data).unwrap()
    }
}

/// Adding to a scalar to a tensors together.
impl<E: Element> Add<E> for TensorImpl<E>
{
    type Output = Self;

    fn add(self, scalar: E) -> Self {
        let data = self
            .data
            .iter()
            // TODO(mhauru) What's the consequence of cloning here? Does it affect performance?
            .map(|a| a.clone() + scalar.clone())
            .collect();
        TensorImpl::from_vec(self.shape(), data)
    }
}


impl<E> Tensor<E> for TensorImpl<E>
where
    E: Element,
{
    type TensorError = Error;

    fn from_vec(shape: Vec<usize>, data: Vec<E>) -> Result<Self, Self::TensorError> {
        if shape.iter().product::<usize>() != data.len() {
            return Err(Error::msg(
                "The length of the `data` param does not match the values of the `shape` param",
            ));
        } else {
            Ok(TensorImpl { shape, data })
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

    //fn transpose(self) -> Self;

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
        let maybe_tensor = TensorImpl::from_vec(shape.clone(), data.clone());
        let tensor = maybe_tensor.unwrap();

        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.data, data);
    }

    #[test]
    fn test_from_vec_invalid_params() {
        // The length of the `data` Vec is not that implied by the `shape` Vec
        let shape = vec![2, 3];
        let data = vec![1, 2];
        let maybe_tensor = TensorImpl::from_vec(shape.clone(), data.clone());
        assert!(maybe_tensor.is_err());
        let err = maybe_tensor.unwrap_err();

        assert!(
            err.to_string().contains(
                "The length of the `data` param does not match the values of the `shape` param"
            )
        );
    }


    #[test]
    fn test_shape() {
        let shape = vec![2, 3];
        let data = vec![1, 2, 3, 4, 5, 6];
        let tensor = TensorImpl::from_vec(shape.clone(), data);

        assert_eq!(tensor.unwrap().shape(), shape);
    }
}

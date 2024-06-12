use anyhow::Error;
use interfaces::tensors::{Element, RealElement, RealTensor, Tensor};
use interfaces::utils::{Exp, Ln, Pow};
use std::{
    fmt::Debug,
    ops::{Add, Mul},
    vec::Vec,
};

/// Implementation of multidimensional arrays as row major strided vectors.
#[derive(Debug, Clone, PartialEq)]
struct TensorImpl<E>
where
    E: Element,
{
    shape: Vec<usize>,
    data: Vec<E>,
}

fn strides_from_shape(shape: &Vec<usize>) -> Vec<usize> {
    let mut strides = vec![1];
    for i in 0..(shape.len() - 1) {
        strides.push(strides[i] * shape[shape.len() - 1 - i]);
    }
    return strides.into_iter().rev().collect();
}

fn num_elements_from_shape(shape: &Vec<usize>) -> usize {
    return shape.iter().product::<usize>();
}

impl<E: Element> TensorImpl<E> {
    fn strides(&self) -> Vec<usize> {
        return strides_from_shape(&self.shape);
    }

    fn num_dims(&self) -> usize {
        return self.shape.len();
    }

    fn num_elements(&self) -> usize {
        return self.data.len();
    }

    fn vec_indx(&self, idxs: Vec<usize>) -> usize {
        let s = self.strides();
        return idxs
            .iter()
            .zip(s.iter())
            .fold(0, |acc, (idx, stride)| acc + idx * stride);
    }

    // TODO(mhauru) This should return something like a view, which references the same data but is
    // a new object. I don't know how to do that though.
    //fn reshape(&mut self, new_shape: Vec<usize>) {
    //    if self.num_elements() != num_elements_from_shape(&new_shape) {
    //        panic!("The number of elements in the new shape does not match the number of elements in the original shape.");
    //    }
    //    self.shape = new_shape;
    //}

    /// Multiply the tensor by the transpose of a matrix.
    ///
    /// This is equivalent to `self.matmul(other.transpose())`, but faster.
    fn matmul_transpose(
        &self,
        other: &Self,
    ) -> Result<TensorImpl<E>, <TensorImpl<E> as Tensor<E>>::TensorError> {
        let self_num_dims = self.num_dims();
        let other_num_dims = other.num_dims();
        if self_num_dims < 2 {
            return Err(Error::msg(
                "The first tensor int matmul must have at least 2 dimensions",
            ));
        }
        if other_num_dims != 2 {
            return Err(Error::msg(
                "The second tensor int matmul must have 2 dimensions",
            ));
        }
        let dim1 = self.shape[self_num_dims - 2];
        let dim_inner = self.shape[self_num_dims - 1];
        let dim2 = other.shape[other_num_dims - 2];
        if dim_inner != other.shape[other_num_dims - 1] {
            return Err(Error::msg(
                "The contracted dimensions of the tensors must match.",
            ));
        }
        let lead_dim = self
            .shape
            .iter()
            .take(self_num_dims - 2)
            .into_iter()
            .product::<usize>();

        // Create an unallocated data vector for the result.
        let n_elements = lead_dim * dim1 * dim2;
        let mut new_data: Vec<E> = Vec::with_capacity(n_elements);
        let mut new_shape: Vec<usize> =
            self.shape.iter().take(self_num_dims - 2).cloned().collect();
        new_shape.push(dim1);
        new_shape.push(dim2);

        // Loop over the elements in the order of new_data.
        let self_strides = self.strides();
        let other_strides = other.strides();
        let lead_stride = if self_num_dims > 2 {
            self_strides[self_num_dims - 3]
        } else {
            1
        };
        for i in 0..lead_dim {
            for j1 in 0..dim1 {
                for j2 in 0..dim2 {
                    let mut accumulator = E::zero();
                    for j_inner in 0..dim_inner {
                        let self_idx = i * lead_stride
                            + j1 * self_strides[self_num_dims - 2]
                            + j_inner * self_strides[self_num_dims - 1];
                        let other_idx = j2 * other_strides[other_num_dims - 2]
                            + j_inner * other_strides[other_num_dims - 1];
                        accumulator += self.data[self_idx].clone() * other.data[other_idx].clone();
                    }
                    new_data.push(accumulator);
                }
            }
        }
        return Result::Ok(TensorImpl {
            shape: new_shape,
            data: new_data,
        });
    }
}

impl<E: Element> IntoIterator for TensorImpl<E> {
    type Item = E;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> std::vec::IntoIter<Self::Item> {
        self.data.into_iter()
    }
}

impl<E: Element> From<TensorImpl<E>> for Vec<E> {
    fn from(value: TensorImpl<E>) -> Self {
        value.data.into_iter().collect()
    }
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
        if num_elements_from_shape(shape) != data.len() {
            return Err(Error::msg(
                "The length of the `data` param does not match the values of the `shape` param",
            ));
        } else {
            Ok(TensorImpl {
                shape: shape.clone(),
                data: data.clone(),
            })
        }
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// Fill a matrix by repeatedly cloning the provided element.
    /// Note: the behaviour might be unexpected if the provided element clones "by reference".
    fn fill_with_clone(shape: Vec<usize>, element: E) -> Self {
        let data = vec![element; num_elements_from_shape(&shape)];
        TensorImpl { shape, data }
    }

    fn at(&self, idxs: Vec<usize>) -> Option<&E> {
        return self.data.get(self.vec_indx(idxs));
    }

    fn at_mut(&mut self, idxs: Vec<usize>) -> Option<&mut E> {
        let index = self.vec_indx(idxs);
        return self.data.get_mut(index);
    }

    fn transpose(&self) -> Self {
        if self.shape.len() < 2 {
            return self.clone();
        }
        let mut new_shape = self.shape.clone();
        let num_dims = self.num_dims();
        // Swap the last two elements of the shape vector
        new_shape.swap(num_dims - 1, num_dims - 2);
        // Compute the product of all the dimensions _not_ being transposed.
        let lead_dim = new_shape
            .iter()
            .take(num_dims - 2)
            .into_iter()
            .product::<usize>();

        // Create an unallocated data vector the same size as the original.
        let n_elements = self.num_elements();
        let mut new_data: Vec<E> = Vec::with_capacity(n_elements);

        // Loop over the elements in the order of new_data.
        let strides = self.strides();
        let lead_stride = if num_dims > 2 {
            strides[num_dims - 3]
        } else {
            1
        };
        for i in 0..lead_dim {
            for j in 0..new_shape[num_dims - 2] {
                for k in 0..new_shape[num_dims - 1] {
                    // Find the data index of this element in the original matrix.
                    // Note the reversal of the roles of k an j.
                    let transposed_idx =
                        i * lead_stride + k * strides[num_dims - 2] + j * strides[num_dims - 1];
                    new_data.push(self.data[transposed_idx].clone());
                }
            }
        }
        return TensorImpl {
            shape: new_shape,
            data: new_data,
        };
    }

    fn matmul(&self, other: &Self) -> Result<Self, Self::TensorError> {
        return self.matmul_transpose(&other.transpose());
    }

    /// Interleaves last dimension of the input tensors and concatenates them along the last dimension.
    fn concat(
        self: &Self,
        other: &Self,
        dim: usize,
    ) -> Result<Self, <Self as Tensor<E>>::TensorError> {
        if self.num_dims() != other.num_dims() {
            return Err(Error::msg(
                "Tensors must have the same number of dimensions",
            ));
        }
        if dim != self.num_dims() - 1 {
            return Err(Error::msg(
                "Concatenation is only supported along the last dimension",
            ));
        }
        for i in 0..self.shape.len() {
            if i != dim && self.shape[i] != other.shape[i] {
                return Err(Error::msg(
                    "Tensors must have the same shape except for the concatenation dimension",
                ));
            }
        }

        let self_last_dim = *self.shape.last().unwrap();
        let other_last_dim = *other.shape.last().unwrap();

        // Interleave the two flattened tensors in chunks equal to size of last dim of respective tensors
        let new_data: Vec<E> = self
            .data
            .chunks(self_last_dim)
            .zip(other.data.chunks(other_last_dim)) // yields items like (&[1.0, 2.0, 3.0], &[7.0, 8.0, 9.0])
            .flat_map(|(a, b)| a.into_iter().chain(b)) // chains to produce iterators like [1.0, 2.0, 3.0, 7.0, 8.0, 9.0]
            // TODO: consider adding a bound on copy
            // .copied()
            .cloned() // &f64 -> f64, optional
            .collect();

        // Convert output_vec into a tensor with required output shape
        let mut new_shape = self.shape.clone();
        new_shape[dim] = self.shape[dim] + other.shape[dim];
        return Ok(Self {
            shape: new_shape,
            data: new_data,
        });
    }

    /// Sum across one or more dimensions (eg. row-wise sum for a 2D matrix resulting in a "column
    /// vector")
    fn dim_sum(&self, dims: Vec<usize>) -> Self {
        // naive implementation, just looping over the dimensions
        let mut result = self.clone();
        for dim in dims {
            result = result.single_dim_sum(dim);
        }
        return result;
    }
}

impl<E> TensorImpl<E>
where
    E: Element,
{
    ///// Sum across a single dimensions (eg. row-wise sum for a 2D matrix resulting in a "column
    ///// vector")
    fn single_dim_sum(&self, dim: usize) -> Self {
        if dim >= self.shape.len() {
            panic!("The provided dimension is out of bounds.");
        }

        let leading_dims = self.shape.iter().take(dim).into_iter().product::<usize>();
        let trailing_dims = self
            .shape
            .iter()
            .skip(dim + 1)
            .into_iter()
            .product::<usize>();

        println!("leading_dims: {}", leading_dims);
        println!("self.shape[dim]: {}", self.shape[dim]);
        println!("trailing_dims: {}", trailing_dims);

        let mut output_shape = self.shape.clone();
        output_shape[dim] = 1;
        let output_size = output_shape.iter().product::<usize>();

        let mut dim_sum: Vec<E> = Vec::new();

        // Outer loop needs to iterate over the size of the new shape
        for lead_idx in 0..leading_dims {
            for trail_idx in 0..trailing_dims {
                let mut sum: E = E::zero();
                for summing_idx in 0..self.shape[dim] {
                    let idx = lead_idx * self.shape[dim] * trailing_dims
                        + summing_idx * trailing_dims
                        + trail_idx;
                    // let idx = summing_idx + lead_idx + (trail_idx * leading_dims);

                    // sum += original[lead_idx,summing_idx,trail_idx]
                    sum += self.data[idx].clone();
                }
                // for (j, value) in self.data.iter().enumerate() {
                //     // print value of i, j and value
                //     let a = j % leading_dims;
                //     let b = j % trailing_dims;

                //     if j % self.shape[dim] == trail_idx {
                //         sum += value.clone();
                //     }
                //     println!("i: {}, j: {}, a: {}, b: {}, value: {}, sum: {}", trail_idx, j, a, b, value, sum);
                // }
                dim_sum.push(sum);
            }
        }

        TensorImpl {
            shape: output_shape,
            data: dim_sum,
        }
    }
}

impl<E: RealElement> Exp for TensorImpl<E> {
    fn exp(self) -> Self {
        todo!()
    }
}

impl<E: RealElement> Pow<E> for TensorImpl<E> {
    fn pow(self, exp: E) -> Self {
        todo!()
    }
}

impl<E: RealElement> Ln for TensorImpl<E> {
    fn ln(self) -> Self {
        todo!()
    }
}

impl<E: RealElement> RealTensor<E> for TensorImpl<E> {
    fn softmax(&self, dim: usize) -> Self {
        todo!()
    }

    fn fill_from_f64(shape: Vec<usize>, data: f64) -> Self {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

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
        // Transposing twice should return the original tensor
        let transposed_twice1 = transposed1.transpose();
        assert_eq!(transposed_twice1.shape(), shape1);
        assert_eq!(transposed_twice1.data, data1);

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

        // The data should be different from the original tensor
        assert_ne!(transposed.data, original_data);

        // Transposing twice should return the original tensor
        let transposed_twice = transposed.transpose();
        assert_eq!(transposed_twice.shape(), shape);
        assert_eq!(transposed_twice.data, original_data);
    }

    #[test]
    fn test_at() {
        let shape = vec![3, 2, 2];
        let original_data = (1..13).collect::<Vec<i32>>();
        let tensor = TensorImpl::from_vec(&shape, &original_data).unwrap();

        assert_eq!(*tensor.at(vec![0, 0, 0]).unwrap(), 1);
        assert_eq!(*tensor.at(vec![2, 1, 1]).unwrap(), 12);
        assert_eq!(tensor.at(vec![2, 1, 2]), None);
        assert_eq!(*tensor.at(vec![1, 1, 0]).unwrap(), 7);
        assert_eq!(*tensor.at(vec![1, 0, 1]).unwrap(), 6);
    }

    #[test]
    fn test_at_mut() {
        let shape = vec![3, 2, 2];
        let original_data = (1..13).collect::<Vec<i32>>();
        let mut tensor = TensorImpl::from_vec(&shape, &original_data).unwrap();

        let index = vec![2, 0, 1];
        let element = tensor.at_mut(index.clone()).unwrap();
        *element = 100;
        assert_eq!(*tensor.at(index.clone()).unwrap(), 100);
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

    #[test]
    fn test_single_dim_sum() {
        let shape = vec![2, 2, 2];
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let tensor = TensorImpl::from_vec(&shape, &data).unwrap();

        let expected_depth_sum = vec![6, 8, 10, 12];
        let expected_depth_shape = vec![1, 2, 2];
        let actual_depth_sum = tensor.single_dim_sum(0);
        assert_eq!(actual_depth_sum.data, expected_depth_sum);
        assert_eq!(actual_depth_sum.shape, expected_depth_shape);

        let expected_col_sum = vec![4, 6, 12, 14];
        let expected_col_shape = vec![2, 1, 2];
        let actual_col_sum = tensor.single_dim_sum(1);
        assert_eq!(actual_col_sum.data, expected_col_sum);
        assert_eq!(actual_col_sum.shape, expected_col_shape);

        let expected_row_sum = vec![3, 7, 11, 15];
        let expected_row_shape = vec![2, 2, 1];
        let actual_row_sum = tensor.single_dim_sum(2);
        assert_eq!(actual_row_sum.data, expected_row_sum);
        assert_eq!(actual_row_sum.shape, expected_row_shape);
    }

    #[test]
    fn test_multiple_dim_sum() {
        let original_shape = vec![2, 3, 4, 5];
        let original_data_len = original_shape.iter().product::<usize>();
        let opiginal_data = (1..original_data_len + 1).collect::<Vec<usize>>();

        let tensor = TensorImpl::from_vec(&original_shape, &opiginal_data).unwrap();

        // Sum over dimensions 1 and 3
        // The result should have shape [2, 1, 4, 1]
        // The result should not be dependent of the order of the dimensions
        let expected_shape = vec![2, 1, 4, 1];

        let actual_sum_fwd = tensor.dim_sum(vec![1, 3]);
        let actual_sum_bwd = tensor.dim_sum(vec![3, 1]);

        assert_eq!(actual_sum_fwd.shape, expected_shape);
        assert_eq!(actual_sum_bwd.shape, expected_shape);
        assert_eq!(actual_sum_fwd.data, actual_sum_bwd.data);
    }

    #[test]
    fn test_matmul() {
        let shape1 = vec![3, 2];
        let data1 = vec![
            0.3482904331369514,
            0.6655930709447386,
            0.8870176843052003,
            0.5022740751412192,
            -0.7230098637226343,
            -0.25526520068994935,
        ];
        let tensor1 = TensorImpl::from_vec(&shape1, &data1).unwrap();

        let shape2 = vec![2, 2];
        let data2 = vec![
            -1.7175862448910375,
            -1.5269576121156156,
            0.056288722445149646,
            -0.4679238780318657,
        ];
        let tensor2 = TensorImpl::from_vec(&shape2, &data2).unwrap();

        // The result of multiplying the above two, computed with NumPy.
        let shape_expected = vec![3, 2];
        let data_expected = vec![
            -0.5607534735513462,
            -0.8432716190531137,
            -1.4952570075306946,
            -1.5894644382059397,
            1.2274632448189984,
            1.223450097679374,
        ];

        let result = tensor1.matmul(&tensor2).unwrap();
        assert_eq!(result.shape, shape_expected);
        assert_eq!(result.data, data_expected);

        // Check that A^T * B^T = (B * A)^T
        let shape3 = vec![2, 2];
        let data3 = vec![
            0.7670520197177377,
            0.14770389167309558,
            0.3166885668777075,
            0.326126502112053,
        ];
        let tensor3 = TensorImpl::from_vec(&shape3, &data3).unwrap();

        let lhs = tensor2.transpose().matmul(&tensor3.transpose()).unwrap();
        let rhs = (tensor3.matmul(&tensor2).unwrap()).transpose();
        assert_eq!(lhs, rhs)
    }

    #[test]
    fn test_matmul_multidim() {
        let mut rng = rand::thread_rng();

        let shape1 = vec![5, 4, 3, 2];
        let num_elements1 = num_elements_from_shape(&shape1);
        let data1: Vec<f64> = (0..num_elements1).map(|_| rng.gen::<f64>()).collect();
        let tensor1 = TensorImpl::from_vec(&shape1, &data1).unwrap();

        let shape2 = vec![2, 6];
        let num_elements2 = num_elements_from_shape(&shape2);
        let data2: Vec<f64> = (0..num_elements2).map(|_| rng.gen::<f64>()).collect();
        let tensor2 = TensorImpl::from_vec(&shape2, &data2).unwrap();

        let result = tensor1.matmul(&tensor2).unwrap();
        let shape_expected = vec![5, 4, 3, 6];
        assert_eq!(result.shape, shape_expected);
    }

    #[test]
    fn test_fill_with_clone() {
        let shape = vec![2, 3];
        let element = 10;
        let tensor = TensorImpl::fill_with_clone(shape, element);

        assert_eq!(tensor.data, vec![10, 10, 10, 10, 10, 10]);
    }

    #[test]
    fn test_into_iter() {
        let shape = vec![2, 3];
        let data = vec![1, 2, 3, 4, 5, 6];
        let tensor = TensorImpl::from_vec(&shape, &data).unwrap();
        assert_eq!(tensor.into_iter().collect::<Vec<i32>>(), data);
    }

    #[test]
    fn test_to_vec() {
        let shape = vec![2, 3];
        let data = vec![1, 2, 3, 4, 5, 6];
        let tensor = TensorImpl::from_vec(&shape, &data).unwrap();
        assert_eq!(Vec::<i32>::from(tensor), data);
    }

    #[test]
    fn test_concat_shape() {
        let mut rng = rand::thread_rng();

        let shape1 = vec![5, 4, 3, 2];
        let num_elements1 = num_elements_from_shape(&shape1);
        let data1: Vec<f64> = (0..num_elements1).map(|_| rng.gen::<f64>()).collect();
        let tensor1 = TensorImpl::from_vec(&shape1, &data1).unwrap();

        let shape2 = vec![5, 4, 3, 3];
        let num_elements2 = num_elements_from_shape(&shape2);
        let data2: Vec<f64> = (0..num_elements2).map(|_| rng.gen::<f64>()).collect();
        let tensor2 = TensorImpl::from_vec(&shape2, &data2).unwrap();

        let result = tensor1.concat(&tensor2, 3).unwrap();
        let shape_expected = vec![5, 4, 3, 5];
        assert_eq!(result.shape, shape_expected);
    }

    #[test]
    /// Test that concat(A * B, A * C, 2) == A * concat(B, C, 2).
    fn test_concat_matmul() {
        let mut rng = rand::thread_rng();

        let shape1 = vec![3, 2];
        let num_elements1 = num_elements_from_shape(&shape1);
        let data1: Vec<f64> = (0..num_elements1).map(|_| rng.gen::<f64>()).collect();
        let tensor1 = TensorImpl::from_vec(&shape1, &data1).unwrap();

        let shape2 = vec![3, 4];
        let num_elements2 = num_elements_from_shape(&shape2);
        let data2: Vec<f64> = (0..num_elements2).map(|_| rng.gen::<f64>()).collect();
        let tensor2 = TensorImpl::from_vec(&shape2, &data2).unwrap();

        let shape3 = vec![3, 3];
        let num_elements3 = num_elements_from_shape(&shape3);
        let data3: Vec<f64> = (0..num_elements3).map(|_| rng.gen::<f64>()).collect();
        let tensor3 = TensorImpl::from_vec(&shape3, &data3).unwrap();

        let result1 = tensor3
            .matmul(&tensor1.concat(&tensor2, 1).unwrap())
            .unwrap();
        let result2 = tensor3
            .matmul(&tensor1)
            .unwrap()
            .concat(&tensor3.matmul(&tensor2).unwrap(), 1)
            .unwrap();
        assert_eq!(result1.shape, result2.shape);
        assert_eq!(result1.data, result2.data);
    }
}

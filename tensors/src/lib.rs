use interfaces::tensors::Tensor;

struct TensorImpl<E> {
    shape: Vec<usize>,
    data: Vec<E>,
}

impl<E> Tensor<E> for TensorImpl<E> {
    fn new(shape: Vec<usize>, data: Vec<E>) -> Self {
        TensorImpl {
            shape,
            data,
        }
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
    }

    fn from_vec(shape: Vec<usize>, data: Vec<E>) -> Self;

    /// Fill a matrix by repeatedly cloning the provided element.
    /// Note: the behaviour might be unexpected if the provided element clones "by reference".
    fn fill_with_clone(shape: Vec<usize>, element: E) -> Self;

    fn at(&self, idxs: Vec<usize>) -> Option<&E>;

    fn at_mut(&mut self, idxs: Vec<usize>) -> Option<&mut E>;

    fn transpose(self) -> Self;

    fn matmul(&self, other: &Self) -> Result<Self, Self::TensorError>;

    /// Sum across one or more dimensions (eg. row-wise sum for a 2D matrix resulting in a "column
    /// vector")
    fn dim_sum(&self, dim: Vec<usize>) -> Self;
}

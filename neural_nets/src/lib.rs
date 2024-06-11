use interfaces::deep_learning::{DLModule, LinearLayer};
use interfaces::tensors::Element;
use interfaces::tensors::Tensor;
use std::marker::PhantomData;
pub mod embedding_table;

struct LinLayer<T: Tensor<E>, E: Element> {
    w: T,
    b: T,
    tensor_element_phantom: PhantomData<E>,
}

impl<T, E> DLModule<T, E> for LinLayer<T, E>
where
    T: Tensor<E>,
    E: Element,
{
    type DLModuleError = <T as Tensor<E>>::TensorError;
    fn forward(&self, x: &T) -> Result<T, Self::DLModuleError> {
        Ok(self.w.matmul(&x.clone().transpose())? + self.b.clone())
    }
    fn params(&self) -> Vec<E> {
        let mut res: Vec<E> = self.w.clone().into();
        res.extend(self.b.clone().into());
        res
    }
}

impl<T, E> LinearLayer<T, E> for LinLayer<T, E>
where
    T: Tensor<E>,
    E: Element,
{
}

impl<T, E> LinLayer<T, E>
where
    T: Tensor<E>,
    E: Element + From<f64>,
{
    fn new(i_size: usize, o_size: usize) -> Self {
        let weights = T::from_vec(&vec![o_size, i_size], data)
            .expect("Ensured data can be arranged into a matrix of the given size.");
        let bias = T::from_vec(&vec![o_size, 1_usize], data)
            .expect("Ensured data can be arranged into a matrix of the given size.");

        LinLayer {
            w: weights,
            b: bias,
            tensor_element_phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]s
}

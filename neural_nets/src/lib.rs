use anyhow;
use interfaces::deep_learning::{DLModule, LinearLayer};
use interfaces::tensors::Element;
use interfaces::tensors::Tensor;
use std::marker::PhantomData;

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
        todo!()
    }
    fn params(&self) -> Vec<E> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn it_works() {
    //     let result = add(2, 2);
    //     assert_eq!(result, 4);
    // }
}

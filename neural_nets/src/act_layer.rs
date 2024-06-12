// This module contains the activation layer struct and its implementation.
use interfaces::{
    deep_learning::{ActivationLayer, DLModule},
    tensors::{Element, Tensor},
};
use std::marker::PhantomData;

struct ActLayer<T: Tensor<E>, E: Element> {
    placeholder: T,
    tensor_element_phantom: PhantomData<E>,
}

impl<T, E> DLModule<T, E> for ActLayer<T, E>
where
    T: Tensor<E>,
    E: Element,
{
    type DLModuleError = <T as Tensor<E>>::TensorError;

    fn forward(&self, z: &T) -> Result<T, Self::DLModuleError> {
        todo!("Implement the forward method for the activation layer");
    }

    fn params(&self) -> Vec<E> {
        todo!("Implement the params method for the activation layer")
    }
}

impl<T, E> ActivationLayer<T, E> for ActLayer<T, E>
where
    T: Tensor<E>,
    E: Element,
{
}

impl<T, E> ActLayer<T, E>
where
    T: Tensor<E>,
    E: Element,
{
    fn new() -> Self {
        todo!("Implement the new method for the activation layer")
    }
}

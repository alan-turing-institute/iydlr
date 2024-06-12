// This module contains the activation layer struct and its implementation.
use interfaces::{
    deep_learning::{ActivationLayer, DLModule},
    tensors::{Element, Tensor},
};
use std::iter::Iterator;
use std::marker::PhantomData;

struct ActLayer<T: Tensor<E>, E: Element> {
    tensor_phantom: PhantomData<T>,
    tensor_element_phantom: PhantomData<E>,
}

impl<T, E> DLModule<T, E> for ActLayer<T, E>
where
    T: Tensor<E>,
    E: Element + Into<f64>,
{
    type DLModuleError = <T as Tensor<E>>::TensorError;

    fn forward(&self, x: &T) -> Result<T, Self::DLModuleError> {
        let input_shape = x.shape();
        // The shape of the input tensor must be (B, T, C)
        if input_shape.len() != 3 {
            return Err(
                anyhow::Error::msg("The shape of the input tensor must be (B, T, C)").into(),
            );
        } else {
            // The activation function is the ReLU function
            let relu: Vec<_> = x
                .clone()
                .into_iter()
                .map(|x| if x.clone().into() > 0.0 { x } else { E::zero() })
                .collect();
            return Ok(T::from_vec(&input_shape, &relu)?);
        }
    }

    // The activation layer has no parameters, return an empty vector
    fn params(&self) -> Vec<E> {
        Vec::new()
    }
}

impl<T, E> ActivationLayer<T, E> for ActLayer<T, E>
where
    T: Tensor<E>,
    E: Element + Into<f64>,
{
}

impl<T, E> ActLayer<T, E>
where
    T: Tensor<E>,
    E: Element,
{
    fn new() -> Self {
        ActLayer {
            tensor_phantom: PhantomData,
            tensor_element_phantom: PhantomData,
        }
    }
}

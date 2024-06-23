// This module contains the activation layer struct and its implementation.
use interfaces::{
    deep_learning::{ActivationLayer, DLModule},
    tensors::{Element, Tensor},
};
use std::iter::Iterator;
use std::marker::PhantomData;

pub struct ActLayer<T: Tensor<E>, E: Element> {
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
    fn params(&self) -> Vec<T> {
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
    pub fn new() -> Self {
        ActLayer {
            tensor_phantom: PhantomData,
            tensor_element_phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensors::TensorImpl;

    #[test]
    fn construct_act_layer() {
        // The activation layer is constructed without any errors
        let layer: ActLayer<TensorImpl<f64>, f64> = ActLayer::new();
    }

    #[test]
    fn three_dim_forward() {
        // The forward method of the activation layer is called without any errors
        let layer: ActLayer<TensorImpl<f64>, f64> = ActLayer::new();
        let x = TensorImpl::from_vec(&vec![2, 2, 2], &vec![6.0; 8]).unwrap();
        println!("{:?}", x.shape());
        let out = layer.forward(&x).unwrap();
        assert_eq!(out.shape(), vec![2, 2, 2]);
    }

    #[test]
    fn no_negative_values() {
        // The ReLU function is applied to the input tensor
        let layer: ActLayer<TensorImpl<f64>, f64> = ActLayer::new();
        let x = TensorImpl::from_vec(&vec![1, 2, 2], &vec![-2.0, -1.0, 0.0, 1.0]).unwrap();
        let out = layer.forward(&x).unwrap();
        assert_eq!(
            out,
            TensorImpl::from_vec(&vec![1, 2, 2], &vec![0.0, 0.0, 0.0, 1.0]).unwrap()
        )
    }
}

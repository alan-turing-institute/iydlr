// This module contains the activation layer struct and its implementation.
use interfaces::{
    deep_learning::{DLModule, LinearLayer},
    tensors::{Element, RealElement, RealTensor, Tensor},
};
use std::marker::PhantomData;

pub struct NormLayer<T: Tensor<E>, E: Element> {
    tensor_phantom: PhantomData<T>,
    tensor_element_phantom: PhantomData<E>,
}

impl<T, E> DLModule<T, E> for NormLayer<T, E>
where
    T: RealTensor<E>,
    E: RealElement + Into<f64>,
{
    type DLModuleError = <T as Tensor<E>>::TensorError;

    // fn forward(&self, x: &T) -> Result<T, Self::DLModuleError> {
    //     let shape = x.shape();
    //     let sum = x.dim_sum(vec![shape.len() - 1]);
    //     //let mean = x.clone() / sum;
    //     let mean = x.clone() / sum;
    //     let diff: T = x.clone() - mean.clone();
    //     let diff_squared = diff.clone() * diff.clone();
    //     let sd = (diff_squared.clone()
    //         / (diff_squared.dim_sum(vec![shape.len() - 1]) + E::from(f64::EPSILON)))
    //     .pow(E::from(0.5));

    //     Ok(diff / sd)
    // }

    fn forward(&self, x: &T) -> Result<T, Self::DLModuleError> {
        // let t_small = E::from(f64::EPSILON);
        // let n_dims = x.shape().len();
        // let sum = x.dim_sum(vec![n_dims - 1]);
        // let mean = x.clone() / (sum + t_small.clone());
        // let diff: T = x.clone() - mean.clone();
        // let diff_squared = diff.clone() * diff.clone();
        // let diff_squared_sum = diff_squared.dim_sum(vec![n_dims - 1]);
        // let sd = (diff_squared.clone() / (diff_squared_sum + t_small.clone()) + t_small.clone())
        //     .pow(E::from(0.5));

        // Ok(diff / sd)
        let final_dim_idx = x.shape().len() - 1;
        let final_dim_size = E::from(*x.shape().last().unwrap() as f64);
        let epsilon = E::from(f64::EPSILON);

        // B,T
        let mean = x.dim_sum(vec![final_dim_idx]) / final_dim_size.clone();

        // B,T,C
        let numerator = x.clone() - mean.clone();

        // B,T
        let var = (numerator.clone())
            .pow(E::from(2.0))
            .dim_sum(vec![final_dim_idx])
            / final_dim_size;

        let denominator = (var + epsilon).pow(E::from(0.5));

        Ok(numerator / denominator)
    }

    // The activation layer has no parameters, return an empty vector
    fn params(&self) -> Vec<T> {
        Vec::new()
    }
}

impl<T, E> LinearLayer<T, E> for NormLayer<T, E>
where
    T: RealTensor<E>,
    E: RealElement + Into<f64>,
{
}

impl<T, E> NormLayer<T, E>
where
    T: Tensor<E>,
    E: Element,
{
    pub fn new() -> Self {
        Self {
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
    fn layer_norm() {
        let x = TensorImpl::from_vec(&vec![1, 2, 3], &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let layer = NormLayer::new();
        let y = layer.forward(&x);
        println!("{:?}", y);
    }
}

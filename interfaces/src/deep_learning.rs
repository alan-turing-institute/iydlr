use crate::tensors::{Element, Tensor};
use std::fmt::Debug;

/// Deep Learning Module, generic over a Tensor object T and it's elements E.
pub trait DLModule<T, E>
where
    T: Tensor<E>,
    E: Element,
{
    type DLModuleError: Debug
        + From<<T as Tensor<E>>::TensorError>
        + Into<<T as Tensor<E>>::TensorError>;

    /// Forward pass through the module. Implementers must support a 1-dimensional `x` and may also
    /// support forwarding an n-dimensional `x` (eg. a "batch").
    fn forward(&self, x: &T) -> Result<T, Self::DLModuleError>;

    fn params(&self) -> Vec<E>;
}

/// A convenince-only Subtrait of `DLModule` to specifiy a module that does linear transformation.
/// There is an expectation that `.forward()` will preserve the number of dimensions of the input `x`
/// (unlike `EmbeddingLayer.forward()`).
pub trait LinearLayer<T, E>: DLModule<T, E>
where
    T: Tensor<E>,
    E: Element,
{
}

/// A convenince-only Subtrait of `DLModule` to specifiy a module that does non-linear transformation.
/// There is an expectation that `.forward()` will preserve the number of dimensions of the input `x`
/// (unlike `EmbeddingLayer.forward()`).
pub trait ActivationLayer<T, E>: DLModule<T, E>
where
    T: Tensor<E>,
    E: Element,
{
}

/// A convenince-only Subtrait of `DLModule` to specifiy an embedding module. Whilst the
/// implementation of `EmbeddingLayer.forward()` may look very similar to `LinearLayer.forward()`, the
/// expectation is that the input to `EmbeddingLayer.forward()` will be a Tensor of indicies, each
/// mapping to an embedding vector. Therefore `EmbeddingLayer.forward()` will increase the
/// dimensionality of the input `x` by 1.
pub trait EmbeddingLayer<T, E>: DLModule<T, E>
where
    T: Tensor<E>,
    E: Element,
{
}

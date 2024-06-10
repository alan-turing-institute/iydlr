use std::marker::PhantomData;

use interfaces::deep_learning::{DLModule, LinearLayer};
use interfaces::tensors::{Element, Tensor};

pub trait MaskedSelfAttention<T, E>: DLModule<T, E>
where
    T: Tensor<E>,
    E: Element,
{
}

/// Attention module, generic over the type of the elements contained within the tensors.
/// <script type="math/tex; mode=display">
/// Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
/// </script>
/// `Q,K,V dims`: (batch_size, seq_len, d_k)
pub struct AttentionBlock<T, E, L>
where
    L: LinearLayer<T, E>,
    T: Tensor<E>,
    E: Element,
{
    pub query_weights: L,
    pub key_weights: L,
    pub value_weights: L,
    pub _marker_t: PhantomData<T>,
    pub _marker_e: PhantomData<E>,
}

impl<T, E, L> AttentionBlock<T, E, L>
where
    L: LinearLayer<T, E>,
    T: Tensor<E>,
    E: Element,
{
    pub fn new() -> Self {
        // Generate weights tensors W_Q, W_K, W_V with shapes like X (embedding matrix)
        todo!()
    }
}

// TODO: consider renaming as `LearnableTransform`
impl<T, E, L> DLModule<T, E> for AttentionBlock<T, E, L>
where
    L: LinearLayer<T, E>,
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

impl<T, E, L> MaskedSelfAttention<T, E> for AttentionBlock<T, E, L>
where
    L: LinearLayer<T, E>,
    T: Tensor<E>,
    E: Element,
{
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_construct() {
        todo!()
    }

    #[test]
    fn test_forward() {
        todo!()
    }
}

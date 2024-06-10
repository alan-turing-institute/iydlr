use anyhow::Error;
use interfaces::deep_learning::{DLModule, LinearLayer};
use interfaces::tensors::{Element, Tensor};
use std::marker::PhantomData;

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
    pub mask: T,
    pub _marker_t: PhantomData<T>,
    pub _marker_e: PhantomData<E>,
}

impl<T, E, L> AttentionBlock<T, E, L>
where
    L: LinearLayer<T, E>,
    T: Tensor<E>,
    E: Element,
{
    pub fn new(embedding_dims: (usize, usize)) -> Self {
        // Generate weights tensors W_Q, W_K, W_V with shapes like X (embedding matrix)

        // Assume T has shape: (batch x sequence x channel)
        // X.size
        // let W_Q = L::new();
        // let W_K = L::new();
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
        // todo!()
        // let masked_x: T = self.mask.forward(x)?;
        let masked_x: T = self.mask * x.clone(); // element-wise multiplication

        let query = self.query_weights.forward(x).unwrap(); // just a matmul, Unwrap used since we currently do not have conversion implemented
        let key: T = self.key_weights.forward(x).unwrap();
        let value: T = self.value_weights.forward(x).unwrap();
        let last_dim_of_keys = key.shape().last().unwrap();
        // let last_dim_of_keys = key.shape().last().ok_or(anyhow!("Empty dim"))?;
        // let att: T = query.matmul(&key.transpose()) * 1 / sqrtf64(last_dim_of_keys)?; // make sure only last two dimensions are transposed
        // let att: T = query.matmul(&key.transpose()).unwrap() * 1. / E::sqrt(last_dim_of_keys);
        let att: T = query.matmul(&key.transpose()).unwrap() * T::from_f64(1.)
            / T::from_usize(*last_dim_of_keys);
        // matmul with V
        let att_v: T = att.matmul(&value);
        // make sure only last two dimensions are transposed
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

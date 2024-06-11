use interfaces::deep_learning::{DLModule, LinearLayer};
use interfaces::tensors::{RealElement, RealTensor, Tensor};
use std::marker::PhantomData;

/// Interleaves last dimension of the input tensors and concatenates them along the last dimension.
pub fn concat<E: RealElement, T: RealTensor<E>>(x: T, y: T) -> T {
    let x_shape = x.shape();
    let y_shape = y.shape();
    // TODO: not sure why can't use Vec::from(), but does't compile
    // let v_x: Vec<E> = Vec::<E>::from(x);
    let v_x: Vec<E> = x.into();
    let v_y: Vec<E> = y.into();
    let x_last_dim = *x_shape.last().unwrap();
    let y_last_dim = *y_shape.last().unwrap();

    // Interleave the two flattened tensors in chunks equal to size of last dim of respective tensors
    let output_vec: Vec<E> = v_x
        .chunks(x_last_dim)
        .zip(v_y.chunks(y_last_dim)) // yields items like (&[1.0, 2.0, 3.0], &[7.0, 8.0, 9.0])
        .flat_map(|(a, b)| a.into_iter().chain(b)) // chains to produce iterators like [1.0, 2.0, 3.0, 7.0, 8.0, 9.0]
        // TODO: consider adding a bound on copy
        // .copied()
        .cloned() // &f64 -> f64, optional
        .collect();

    // Convert output_vec into a tensor with required output shape
    T::from_vec(
        &vec![x_shape[0], x_shape[1], x_shape[2] + y_shape[2]],
        &output_vec,
    )
    .unwrap()
}

pub trait MaskedSelfAttention<T, E>: DLModule<T, E>
where
    T: Tensor<E>,
    E: RealElement,
{
}

/// Multi-head attention module, generic over the type of the elements contained within the tensors.
/// <script type="math/tex; mode=display">
/// Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
/// </script>
/// `Q,K,V dims`: (batch_size, seq_len, d_k)
///
/// Currently
pub struct MultiHeadAttention<T, E, L>
where
    L: LinearLayer<T, E>,
    T: Tensor<E>,
    E: RealElement,
{
    pub query_weights: Vec<L>,
    pub key_weights: Vec<L>,
    pub value_weights: Vec<L>,
    pub num_heads: usize,
    pub mask: T,
    pub _marker_t: PhantomData<T>,
    pub _marker_e: PhantomData<E>,
}

impl<T, E, L> MultiHeadAttention<T, E, L>
where
    L: LinearLayer<T, E>,
    T: Tensor<E>,
    E: RealElement,
{
    pub fn new(x: &T, num_heads: usize) -> Self {
        // Generate weights tensors W_Q, W_K, W_V with shapes (embedding_dim, d_k),
        // where d_k is embedding_dim / num_heads. For now, we assume num_heads = 1.
        // Then generate W_Q, W_K, W_V with same shape (batch x sequence x channel)
        let v = x.shape();
        let (batch_size, seq_len, embedding_dim) = (v[0], v[1], v[2]);
        let _d_k = embedding_dim / num_heads;
        let mut mask = Vec::with_capacity(x.shape().into_iter().product());
        let matrix_dim = seq_len;
        for i in 0..batch_size {
            for j in 0..matrix_dim {
                for k in 0..matrix_dim {
                    if k >= j {
                        mask[j * matrix_dim + k] = E::zero();
                    } else {
                        mask[j * matrix_dim + k] = E::neg_inf();
                    }
                }
            }
        }

        let mask = T::from_vec(&vec![batch_size, seq_len, seq_len], &mask);

        // TODO: no constructor currently, come back to when concrete type to implement on
        // let query_weights: Vec<L> = (0..num_heads).map(|_| L::new(embedding_dim, d_k)).collect();
        todo!()
    }
}

// TODO: consider renaming as `LearnableTransform`
impl<T, E, L> DLModule<T, E> for MultiHeadAttention<T, E, L>
where
    L: LinearLayer<T, E>,
    T: RealTensor<E>,
    E: RealElement,
{
    type DLModuleError = <T as Tensor<E>>::TensorError;

    fn forward(&self, x: &T) -> Result<T, Self::DLModuleError> {
        let mut outputs: Vec<T> = vec![];
        let masked_x: T = self.mask.clone() * x.clone(); // element-wise multiplication
        for attention_head_idx in 0..self.num_heads {
            // TODO: fix unwraps once error conversion is handled
            let query = self.query_weights[attention_head_idx].forward(x).unwrap(); // just a matmul, Unwrap used since we currently do not have conversion implemented
            let key: T = self.key_weights[attention_head_idx].forward(x).unwrap();
            let value: T = self.value_weights[attention_head_idx].forward(x).unwrap();
            let last_dim_of_keys = *key.shape().last().unwrap();

            // make sure only last two dimensions are transposed
            let att: T = query.matmul(&key.transpose()).unwrap() *
                // TODO: make this safer
                E::from((last_dim_of_keys as f64).powf(-0.5));

            // softmax along the sequence length, TODO: check correct dim for softmax
            let att_masked_softmax: T = (att * masked_x.clone()).softmax(1);
            // matmul attention masked with V
            let att_v: T = att_masked_softmax.matmul(&value).unwrap();
            outputs.push(att_v);
        }

        // Concatanate over heads
        // TODO: check concat is channel-wise
        Ok(outputs.into_iter().reduce(|acc, x| concat(acc, x)).unwrap())
    }

    fn params(&self) -> Vec<E> {
        todo!()
    }
}

impl<T, E, L> MaskedSelfAttention<T, E> for MultiHeadAttention<T, E, L>
where
    L: LinearLayer<T, E>,
    T: RealTensor<E>,
    E: RealElement,
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
        todo!("When concrete types implemented.")
    }
}

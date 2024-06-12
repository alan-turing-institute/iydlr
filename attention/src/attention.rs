use num_traits::Zero;

use autodiff::node::Node;
use interfaces::deep_learning::{DLModule, LinearLayer};
use interfaces::tensors::{RealElement, RealTensor, Tensor};
use neural_nets::lin_layer::LinLayer;
use tensors::TensorImpl;

use std::marker::PhantomData;

const SEED: u64 = 0;

pub trait SelfAttention<T, E>: DLModule<T, E>
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
#[derive(Debug)]
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
    pub mask: Option<T>,
    pub _marker_t: PhantomData<T>,
    pub _marker_e: PhantomData<E>,
}

type El = Node<f64>;
type Te = TensorImpl<El>;
type La = LinLayer<Te, El>;

#[derive(Debug)]
struct Config {
    batch_size: usize,
    vocab_size: usize,
    seq_len: usize,
    embed_dim: usize,
    num_head: usize,
}

impl MultiHeadAttention<Te, El, La> {
    fn new(config: &Config, is_masked: bool) -> Self {
        // Generate weights tensors W_Q, W_K, W_V with shapes (embedding_dim, d_k),
        // where d_k is embedding_dim / num_heads. For now, we assume num_heads = 1.
        // Then generate W_Q, W_K, W_V with same shape (batch x sequence x channel)
        let embed_dim = config.embed_dim;
        let num_heads = config.num_head;
        let seq_len = config.seq_len;
        // let batch_size = config.batch_size;
        let d_k = config.embed_dim / config.num_head;
        let mask: Option<Te> = if is_masked {
            let mut mask: Vec<Node<f64>> = vec![Node::<f64>::zero(); seq_len * seq_len];
            let matrix_dim = seq_len;
            for j in 0..matrix_dim {
                for k in 0..matrix_dim {
                    if k >= j {
                        mask[j * matrix_dim + k] = Node::<f64>::zero();
                    } else {
                        mask[j * matrix_dim + k] = Node::<f64>::neg_inf()
                    }
                }
            }
            // Some(Te::from_vec(&vec![batch_size, seq_len, seq_len], &mask).unwrap())
            Some(Te::from_vec(&vec![seq_len, seq_len], &mask).unwrap())
        } else {
            None
        };

        // TODO: no constructor currently, come back to when concrete type to implement on
        // let query_weights: Vec<L> = (0..num_heads).map(|_| L::new(embedding_dim, d_k)).collect();

        // Q (B, T, C) * Q_W (B, C, T) = (B, T, T)
        // Q.matmaul(Q_W) : (B x T x C) x (C x T) -> (B x T x T)
        let query_weights: Vec<La> = (0..num_heads)
            .map(|_| La::new(embed_dim, d_k, SEED))
            .collect();
        let value_weights: Vec<La> = (0..num_heads)
            .map(|_| La::new(embed_dim, d_k, SEED))
            .collect();
        let key_weights: Vec<La> = (0..num_heads)
            .map(|_| La::new(embed_dim, d_k, SEED))
            .collect();

        Self {
            query_weights,
            key_weights,
            value_weights,
            num_heads,
            mask,
            _marker_t: PhantomData,
            _marker_e: PhantomData,
        }
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
        for attention_head_idx in 0..self.num_heads {
            let mut batch_outs: Vec<T> = vec![];
            let shape = x.shape();
            // TODO: fix unwraps once error conversion is handled
            let x_batch: Vec<E> = x.clone().into();
            // let x_batch = x_batch.chunks(shape[1] * shape[2]).into_iter().enumerate();
            let x_batch: Vec<Vec<E>> = x_batch
                .chunks(shape[1] * shape[2])
                .map(|chunk| chunk.to_vec())
                .collect();

            for (_, single_batch) in x_batch.into_iter().enumerate() {
                let single_batch_tensor =
                    T::from_vec(&vec![shape[1], shape[2]], &single_batch).unwrap();
                // println!("{:?}", self.query_weights[attention_head_idx]);
                let query = self.query_weights[attention_head_idx]
                    .forward(&single_batch_tensor)
                    .unwrap(); // just a matmul, Unwrap used since we currently do not have conversion implemented
                println!("{:?}", x.shape());
                let key: T = self.key_weights[attention_head_idx]
                    .forward(&single_batch_tensor)
                    .unwrap(); // B x T x d_k (C / num_heads)
                println!("{:?}", key.shape());
                let value: T = self.value_weights[attention_head_idx]
                    .forward(&single_batch_tensor)
                    .unwrap();
                let last_dim_of_keys = *key.shape().last().unwrap(); // d_k
                println!("{:?}", value.shape());
                // make sure only last two dimensions are transposed
                let att: T = query.matmul(&key.transpose()).unwrap() *
                    // TODO: make this safer
                    E::from((last_dim_of_keys as f64).powf(-0.5));

                // softmax along the sequence length, TODO: check correct dim for softmax
                // TODO: mask currently not working with shape
                let att: T = if let Some(mask) = &self.mask {
                    println!("{:?}", mask.shape());
                    println!("{:?}", single_batch_tensor.shape());
                    // element-wise multiplication: (B x T x T)  x (B x T x C)
                    let masked_x: T = mask.clone() * single_batch_tensor.clone();
                    // TODO: why does this not mul?
                    println!("{:?}", masked_x.shape());
                    println!("{:?}", att.shape());
                    att * masked_x
                } else {
                    att
                };
                println!("{:?}", self.mask);
                // TODO: check softmax dim
                // softmax now dim 0 (seq_len)
                // TODO: Softmax currently not implemented
                // let att = att.softmax(0);

                // matmul attention masked with V
                let att_v: T = att.matmul(&value).unwrap();
                batch_outs.push(att_v);
            }

            let outs_shape = batch_outs[0].shape();
            let mut expected_shape = vec![shape[0]];
            expected_shape.extend(outs_shape.into_iter());
            let batch_outs_reshaped_as_single_vec: Vec<E> = batch_outs
                .into_iter()
                .flat_map(|batch_out| batch_out.into())
                .collect();
            let value = T::from_vec(&expected_shape, &batch_outs_reshaped_as_single_vec).unwrap();
            // Reshape att_v to value
            outputs.push(value)
        }

        // Concatanate over heads
        // TODO: check concat is channel-wise
        Ok(outputs
            .into_iter()
            .reduce(|acc, x| acc.concat(&x, acc.shape().len() - 1).unwrap())
            .unwrap())
    }

    fn params(&self) -> Vec<E> {
        todo!()
    }
}

impl<T, E, L> SelfAttention<T, E> for MultiHeadAttention<T, E, L>
where
    L: LinearLayer<T, E>,
    T: RealTensor<E>,
    E: RealElement,
{
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_config() -> Config {
        Config {
            batch_size: 2,
            vocab_size: 10,
            seq_len: 7,
            embed_dim: 20,
            num_head: 4,
        }
    }

    #[test]
    fn test_construct() {
        let config = get_config();
        let attention = MultiHeadAttention::new(&config, true);
        assert_eq!(attention.num_heads, 4);
        assert!(attention.mask.is_some());
        // check that mask has the right shape
        // print the shape of the mask
        //println!("{:?}", attention.mask.as_ref().unwrap().shape());
        assert_eq!(attention.mask.unwrap().shape(), vec![7, 7]);
        assert_eq!(attention.query_weights.len(), 4);
        // println!("{:?}", attention.query_weights[0].w);
        println!("{:?}", attention.key_weights[0].w);
    }

    #[test]
    fn test_forward() {
        let config = get_config();
        let attention = MultiHeadAttention::new(&config, true);

        let x = Te::from_vec(
            &vec![config.batch_size, config.seq_len, config.embed_dim],
            &vec![Node::<f64>::zero(); config.batch_size * config.seq_len * config.embed_dim],
        )
        .unwrap();
        let out = attention.forward(&x).unwrap();
        println!("{:?}", out.shape());
    }
}

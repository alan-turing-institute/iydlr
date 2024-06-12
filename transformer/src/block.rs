use attention::attention::SelfAttention;
use interfaces::{
    deep_learning::{ActivationLayer, DLModule, LinearLayer},
    tensors::{RealElement, Tensor},
};
use std::{marker::PhantomData, process::Output};

// keras_nlp.layers.TransformerEncoder(
//     intermediate_dim,
//     num_heads,
//     dropout=0,
//     activation="relu",
//     layer_norm_epsilon=1e-05,
//     kernel_initializer="glorot_uniform",
//     bias_initializer="zeros",
//     normalize_first=False,
//     **kwargs
// )

/// Expecting input as a Tensor with shape (B x T x C) where:
///   - B is batch size
///   - T is time
///   - C is channel.
/// Example from the keras API ([encoder](https://keras.io/api/keras_nlp/modeling_layers/transformer_encoder/))
struct Block<L, A, T, E, Al>
where
    L: LinearLayer<T, E>,
    A: SelfAttention<T, E>,
    T: Tensor<E>,
    E: RealElement,
    Al: ActivationLayer<T, E>,
{
    pub self_attention: A,
    pub linear_layer1: L, // i: C, o: 4C
    pub activation_layer: Al,
    pub linear_layer2: L, // i: 4C, o: C
    pub intermediate_dim: usize,
    pub num_head: usize,
    pub _marker_t: PhantomData<T>,
    _marker_e: PhantomData<E>,
}

impl<T, E, L, A, Al> DLModule<T, E> for Block<L, A, T, E, Al>
where
    L: LinearLayer<T, E>,
    A: SelfAttention<T, E>,
    T: Tensor<E>,
    E: RealElement,
    Al: ActivationLayer<T, E>,
{
    type DLModuleError = <T as Tensor<E>>::TensorError;

    fn forward(&self, x: &T) -> Result<T, Self::DLModuleError> {
        // A block consists of a self-attention layer followed by a feed-forward neural network.
        // It also implements residual connections after each sub-layer.
        // The residual connection adds the original embedding matrix x to the output of the sub-layer.
        // The feed forward neural network consists of two linear layers with a ReLU activation in between.
        // The first linear layer expands to 4 times the embedding dimension,
        // and the second linear layer projects back to the original embedding dimension.

        // TODO: implement residual connections
        let att: T = self.self_attention.forward(x).unwrap();
        let residual1: T = att.clone() + x.clone();

        let lin: T = self.linear_layer1.forward(&residual1).unwrap();
        let act: T = self.activation_layer.forward(&lin).unwrap();
        let lin2: T = self.linear_layer2.forward(&act).unwrap();

        let residual2: T = lin2.clone() + residual1.clone();

        Ok(residual2)
    }

    fn params(&self) -> Vec<E> {
        todo!()
    }
}

// TODO: once activation is concrete
// impl Block<LinLayer, MultiHeadAttention, TensorImpl, Node<f64>, > {
//     fn new() -> Self {
//         todo!()
//     }
// }

#[cfg(test)]
mod tests {
    #[test]
    fn test_block_constructor() {
        todo!();
    }
    fn test_transformer_constructor() {
        todo!();
    }
}

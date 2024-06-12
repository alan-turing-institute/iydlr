use attention::attention::SelfAttention;
use interfaces::{
    deep_learning::{ActivationLayer, DLModule, LinearLayer},
    tensors::{RealElement, Tensor},
};
use std::{marker::PhantomData, process::Output};

/// Expecting input as a Tensor with shape (B x T x C) where:
///   - B is batch size
///   - T is time
///   - C is channel.
///
struct Block<L, A, T, E, AL>
where
    L: LinearLayer<T, E>,
    A: SelfAttention<T, E>,
    T: Tensor<E>,
    E: RealElement,
    AL: ActivationLayer<T, E>,
{
    pub self_attention: A,
    pub linear_layer1: L, // i: C, o: 4C
    pub activation_layer: AL,
    pub linear_layer2: L, // i: 4C, o: C
    // activation_layer:
    // TODO: make an add layer
    // add: Add,
    _marker_t: PhantomData<T>,
    _marker_e: PhantomData<E>,
}

impl<T, E, L, A, AL> DLModule<T, E> for Block<L, A, T, E, AL>
where
    L: LinearLayer<T, E>,
    A: SelfAttention<T, E>,
    T: Tensor<E>,
    E: RealElement,
    AL: ActivationLayer<T, E>,
{
    type DLModuleError = <T as Tensor<E>>::TensorError;

    fn forward(&self, x: &T) -> Result<T, Self::DLModuleError> {
        // A block consists of a self-attention layer followed by a feed-forward neural network.
        // It also implements residual connections after each sub-layer.
        // The residual connection adds the original embedding matrix x to the output of the sub-layer.
        // The feed forward neural network consists of two linear layers with a ReLU activation in between.
        // The first linear layer expands to 4 times the embedding dimension,
        // and the second linear layer projects back to the original embedding dimension.

        let att: T = self.self_attention.forward(x).unwrap();
        let residual1: T = att.clone() + x.clone();

        let lin: T = self.linear_layer1.forward(&residual1).unwrap();
        let act: T = self.activation_layer.forward(&lin).unwrap();
        let lin2: T = self.linear_layer2.forward(&act).unwrap();

        Ok(lin2)
    }
    //
    // }

    fn params(&self) -> Vec<E> {
        todo!()
    }
}

struct Transformer {}

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

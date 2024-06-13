use std::marker::PhantomData;

use crate::block::Block;
use attention::attention::SelfAttention;
use attention::attention::{El, La, Mal, Te};
use autodiff::node::Node;
use config::Config;
use interfaces::deep_learning::LinearLayer;
use interfaces::deep_learning::{ActivationLayer, DLModule};
use interfaces::tensors::RealTensor;
use interfaces::tensors::{RealElement, Tensor};
use neural_nets::optim::bce;
use neural_nets::{act_layer::ActLayer, lin_layer::LinLayer, optim::OptimSGD, serial::Serial};
use tensors::TensorImpl;

// pub struct Transformer {
//     model: Serial<Te, El>,
// }

pub struct Transformer<L, A, T, E, Al>
where
    L: LinearLayer<T, E>,
    A: SelfAttention<T, E>,
    T: Tensor<E>,
    E: RealElement,
    Al: ActivationLayer<T, E>,
{
    model: Serial<T, E>,
    _marker_l: std::marker::PhantomData<L>,
    _marker_a: std::marker::PhantomData<A>,
    _marker_al: std::marker::PhantomData<Al>,
}

impl Transformer<La, Mal, Te, El, ActLayer<Te, El>> {
    pub fn new(config: &Config) -> Self {
        let mut modules: Vec<
            Box<dyn DLModule<Te, El, DLModuleError = <Te as Tensor<El>>::TensorError>>,
        > = vec![];
        for i in 0..config.num_blocks {
            modules.push(Box::new(Block::new(&config, i == 0)));
        }
        modules.push(Box::new(LinLayer::new(
            config.embed_dim,
            config.vocab_size,
            // config.vocab_size * config.seq_len,
            config.seed,
        )));
        // modules.push(Box::new());
        let model = Serial::new(modules);
        Self {
            model,
            _marker_a: PhantomData,
            _marker_al: PhantomData,
            _marker_l: PhantomData,
        }
    }
}

impl<T, E, L, A, Al> DLModule<T, E> for Transformer<L, A, T, E, Al>
where
    L: LinearLayer<T, E>,
    A: SelfAttention<T, E>,
    T: Tensor<E>,
    E: RealElement,
    Al: ActivationLayer<T, E>,
{
    type DLModuleError = <T as Tensor<E>>::TensorError;

    fn forward(&self, x: &T) -> Result<T, Self::DLModuleError> {
        // return self.model.forward(x);
        todo!()
    }

    fn params(&self) -> Vec<E> {
        // return self.model.params();
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use autodiff::node::Node;
    use num_traits::Zero;

    use super::*;

    fn get_config() -> Config {
        Config {
            batch_size: 2,
            vocab_size: 10,
            seq_len: 7,
            embed_dim: 20,
            num_head: 4,
            num_blocks: 4,
            seed: 0,
        }
    }

    #[test]
    fn test_construct() {
        let config = get_config();
        let model = Transformer::new(&config);
        println!("{}", model.params().len());
    }

    //#[test]
    //fn test_forward() {
    //    let config = get_config();
    //    let model = Transformer<TensorImpl, f64>::new(&config);
    //    let x = TensorImpl<Node<f64>>::from_vec(
    //        &vec![config.batch_size, config.seq_len, config.embed_dim],
    //        &vec![Node::<f64>::zero(); config.batch_size * config.seq_len * config.embed_dim],
    //    )
    //    .unwrap();
    //    let out = model.forward(&x).unwrap();
    //    let expected_shape = vec![2, 7, 20];
    //    let actual_shape = out.shape();
    //    assert_eq!(actual_shape, expected_shape);
    //}
}

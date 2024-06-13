use crate::block::Block;
use attention::attention::{El, La, Mal, Te};
use autodiff::node::Node;
use config::Config;
use interfaces::deep_learning::DLModule;
use interfaces::tensors::RealTensor;
use interfaces::tensors::RealElement;
use neural_nets::optim::bce;
use neural_nets::{act_layer::ActLayer, lin_layer::LinLayer, optim::OptimSGD, serial::Serial};
use tensors::TensorImpl;

pub struct TransformerConfig {
    pub batch_size: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub embed_dim: usize,
    pub num_head: usize,
    pub num_blocks: usize,
    pub seed: u64,
}

pub struct Transformer<T, E>
where
    T: RealTensor<Node<E>>,
    E: RealElement
{
    model: Serial<T, Node<E>>,
}

impl<T, E> Transformer<T, E>
where
    T: RealTensor<Node<E>>,
    E: RealElement
{
    type DLModuleError = <T as Tensor<E>>::TensorError;

pub fn new(config: &TransformerConfig) -> Transformer<T, E> {
    let block_config = Config {
        batch_size: config.batch_size,
        vocab_size: config.vocab_size,
        seq_len: config.seq_len,
        embed_dim: config.embed_dim,
        num_head: config.num_head,
        seed: config.seed,
    };

    let mut modules: Vec<Box<dyn DLModule<T, E, DLModuleError>>> = (0..config.num_blocks)
        .map(|i| Box::new(Block::new(&block_config, i == 0)))
        .collect();
    modules.push(Box::new(LinLayer::new(config.embed_dim, config.vocab_size, config.seed)));
    let model: Serial<T, E> = Serial::new(modules);
    return model;
}
}


#[cfg(test)]
mod tests {
    use autodiff::node::Node;
    use num_traits::Zero;

    use super::*;

    fn get_config() -> TransformerConfig {
        TransformerConfig {
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
        let model = Transformer<TensorImpl, f64>::new(&config);
        println!("{}", model.params().len());
    }

    #[test]
    fn test_forward() {
        let config = get_config();
        let model = Transformer<TensorImpl, f64>::new(&config);
        let x = TensorImpl<Node<f64>>::from_vec(
            &vec![config.batch_size, config.seq_len, config.embed_dim],
            &vec![Node::<f64>::zero(); config.batch_size * config.seq_len * config.embed_dim],
        )
        .unwrap();
        let out = model.forward(&x).unwrap();
        let expected_shape = vec![2, 7, 20];
        let actual_shape = out.shape();
        assert_eq!(actual_shape, expected_shape);
    }
}

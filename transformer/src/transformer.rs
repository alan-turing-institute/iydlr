use crate::block::Block;
use attention::attention::{El, La, Mal, Te};
use autodiff::node::Node;
use config::Config;
use interfaces::deep_learning::DLModule;
use interfaces::tensors::RealElement;
use interfaces::tensors::RealTensor;
use neural_nets::optim::bce;
use neural_nets::{act_layer::ActLayer, lin_layer::LinLayer, optim::OptimSGD, serial::Serial};
use tensors::TensorImpl;

pub struct Transformer {
    model: Serial<Te, El>,
}

impl Transformer {
    pub fn new(config: &Config) -> Transformer {
        let mut modules: Vec<Box<dyn DLModule<Te, El>>> = (0..config.num_blocks)
            .map(|i| Box::new(Block::new(&config, i == 0)))
            .collect();
        modules.push(Box::new(LinLayer::new(
            config.embed_dim,
            config.vocab_size,
            config.seed,
        )));
        let model: Serial<T, E> = Serial::new(modules);
        return model;
    }
}

impl<T, E, L, A, Al> DLModule<T, E> for Transformer {
    type DLModuleError = Te::TensorError;

    fn forward(&self, x: &Te) -> Result<Te, Self::DLModuleError> {
        return self.model.forward(x);
    }

    fn params(&self) -> Vec<El> {
        return self.model.params();
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

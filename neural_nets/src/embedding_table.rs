use interfaces::deep_learning::{DLModule, EmbeddingLayer};
use interfaces::tensors::Element;
use interfaces::tensors::Tensor;
use rand::distributions::Distribution;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use statrs::distribution::Normal;
use std::marker::PhantomData;

struct EmbeddingTable<T: Tensor<E>, E: Element> {
    table: T,
    vocab_size: usize,
    tensor_element_phantom: PhantomData<E>,
}

impl<T, E> DLModule<T, E> for EmbeddingTable<T, E>
where
    T: Tensor<E>,
    E: Element + From<f64> + Into<f64>,
{
    type DLModuleError = <T as Tensor<E>>::TensorError;

    fn forward(&self, x: &T) -> Result<T, Self::DLModuleError> {
        // expect B,T,1
        let x_shape = x.shape();

        if x_shape.len() != 3 || *x_shape.last().expect("dimension of x must be > 0") != 1 {
            return Err(anyhow::Error::msg("Expected input shape to be (B,T,1)").into());
        }

        let mut one_hot_tensor_data = Vec::new();

        for el in x.clone().into_iter() {
            let idx: usize = el.into() as usize;
            let mut one_hot = vec![E::zero(); self.vocab_size];
            one_hot[idx] = E::from(1.0);
            one_hot_tensor_data.extend(one_hot);
        }
        let one_hot_tensor = T::from_vec(
            &vec![x_shape[0], x_shape[1], self.vocab_size],
            &one_hot_tensor_data,
        )?;

        Ok(one_hot_tensor.matmul(&self.table)?)
    }
    fn params(&self) -> Vec<E> {
        self.table.clone().into()
    }
}

impl<T, E> EmbeddingLayer<T, E> for EmbeddingTable<T, E>
where
    T: Tensor<E>,
    E: Element + From<f64> + Into<f64>,
{
}

impl<T, E> EmbeddingTable<T, E>
where
    T: Tensor<E>,
    E: Element + From<f64>,
{
    fn new(n_emb: usize, vocab_size: usize, seed: u64) -> Self {
        // He weight initialisation
        // https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
        let noise_mean = 0.0;
        let noise_std = 1.0;
        let rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(noise_mean, noise_std).unwrap();
        let normal_itr = normal.sample_iter(rng);

        let table_data: Vec<E> = normal_itr.take(n_emb * vocab_size).map(E::from).collect();

        let table = T::from_vec(&vec![vocab_size, n_emb], &table_data)
            .expect("Ensured data can be arranged into a matrix of the given size.");

        EmbeddingTable {
            table,
            vocab_size,
            tensor_element_phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use interfaces::deep_learning::DLModule;
    use std::vec;
    use tensors::TensorImpl;

    use super::*;

    #[test]
    fn construct_embedding_table() {
        let _: EmbeddingTable<TensorImpl<f64>, f64> = EmbeddingTable::new(3, 2, 0);
    }

    #[test]
    fn forward_embedding_table() {
        let table: EmbeddingTable<TensorImpl<f64>, f64> = EmbeddingTable::new(2, 4, 0);

        // x = (B,T,1)
        let x = TensorImpl::from_vec(&vec![4, 6, 1], &vec![3.0; 24]).unwrap();

        // result = (B,T,C)
        let result = table.forward(&x).unwrap();
        assert_eq!(result.shape(), vec![4, 6, 2])
    }
}

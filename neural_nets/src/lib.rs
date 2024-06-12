use interfaces::deep_learning::{DLModule, LinearLayer};
use interfaces::tensors::Element;
use interfaces::tensors::Tensor;
use rand::distributions::Distribution;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use statrs::distribution::Normal;
use std::marker::PhantomData;
// pub mod embedding_table;

struct LinLayer<T: Tensor<E>, E: Element> {
    w: T,
    b: T,
    tensor_element_phantom: PhantomData<E>,
}

impl<T, E> DLModule<T, E> for LinLayer<T, E>
where
    T: Tensor<E>,
    E: Element,
{
    type DLModuleError = <T as Tensor<E>>::TensorError;
    fn forward(&self, x: &T) -> Result<T, Self::DLModuleError> {
        // The addition assumes broadcasting if x is a batch (b,i)
        Ok(self.w.matmul(&x.clone().transpose())? + self.b.clone())
    }
    fn params(&self) -> Vec<E> {
        let mut res: Vec<E> = self.w.clone().into();
        res.extend(self.b.clone().into());
        res
    }
}

impl<T, E> LinearLayer<T, E> for LinLayer<T, E>
where
    T: Tensor<E>,
    E: Element,
{
}

impl<T, E> LinLayer<T, E>
where
    T: Tensor<E>,
    E: Element + From<f64>,
{
    fn new(i_size: usize, o_size: usize, seed: u64) -> Self {
        // He weight initialisation
        // https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
        let noise_mean = 0.0;
        let noise_std = f64::sqrt(2.0 / (i_size as f64));
        let rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(noise_mean, noise_std).unwrap();
        let normal_itr = normal.sample_iter(rng);

        let (w_data, b_data): (Vec<E>, Vec<E>) = normal_itr
            .take(o_size * (i_size + 1))
            .map(E::from)
            .enumerate()
            .fold((Vec::new(), Vec::new()), |mut acc, (idx, el)| {
                if idx < o_size * i_size {
                    acc.0.push(el);
                    acc
                } else {
                    acc.1.push(el);
                    acc
                }
            });

        let weights = T::from_vec(&vec![o_size, i_size], &w_data)
            .expect("Ensured data can be arranged into a matrix of the given size.");
        let bias = T::from_vec(&vec![o_size, 1_usize], &b_data)
            .expect("Ensured data can be arranged into a matrix of the given size.");

        LinLayer {
            w: weights,
            b: bias,
            tensor_element_phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]s
}

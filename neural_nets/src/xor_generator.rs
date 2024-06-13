use std::{collections::HashMap, marker::PhantomData};

use interfaces::tensors::{Element, Tensor};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensors::TensorImpl;

pub struct XorGenerator<E> {
    batch_size: usize,
    rng: ChaCha8Rng,
    number_type_phantom: PhantomData<E>,
}

impl<E> Iterator for XorGenerator<E>
where
    E: Element + From<f64>,
{
    type Item = (TensorImpl<E>, Vec<E>);

    fn next(&mut self) -> Option<Self::Item> {
        Some(single_xor_batch(self.batch_size, &mut self.rng))
    }
}

impl<E> XorGenerator<E> {
    pub fn new(batch_size: usize, seed: u64) -> XorGenerator<E> {
        XorGenerator {
            batch_size,
            rng: ChaCha8Rng::seed_from_u64(seed),
            number_type_phantom: PhantomData,
        }
    }
}

fn single_xor_batch<E>(batch_size: usize, rng: &mut ChaCha8Rng) -> (TensorImpl<E>, Vec<E>)
where
    E: Element + From<f64>,
{
    let x_size = vec![1, batch_size, 2];

    let known_xor = HashMap::from([
        (0, ([0f64, 0f64], 0f64)),
        (1, ([0f64, 1f64], 1f64)),
        (2, ([1f64, 0f64], 1f64)),
        (3, ([1f64, 1f64], 0f64)),
    ]);

    let mut y_result = Vec::<E>::with_capacity(batch_size);
    let mut x_accumulator = Vec::<E>::with_capacity(batch_size * 2);

    for _ in 0..batch_size {
        // Pick a random number between 0 and 3
        let choice = rng.gen_range(0..4);
        let (x, y) = known_xor.get(&choice).unwrap();

        y_result.push((*y).into());
        x_accumulator.extend((*x).into_iter().map(E::from));
    }

    let x_result = TensorImpl::from_vec(&x_size, &x_accumulator).unwrap();

    (x_result, y_result)
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_single_xor_batch() {
        // try this with a few different batch sizes
        for batch_size in [1, 55, 77777] {
            let (x, y) = single_xor_batch::<f64>(batch_size, &mut ChaCha8Rng::seed_from_u64(0));

            assert_eq!(x.shape(), vec![1, batch_size, 2]);
            assert_eq!(y.len(), batch_size);
        }
    }

    #[test]
    fn test_xor_generator() {
        let batch_size = 10;
        let mut generator: XorGenerator<f64> = XorGenerator::new(batch_size, 0);

        for _ in 0..10 {
            let (x, y) = generator.next().unwrap();

            assert_eq!(x.shape(), vec![1, batch_size, 2]);
            assert_eq!(y.len(), batch_size);
        }
    }
}

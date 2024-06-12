use std::collections::HashMap;

use interfaces::tensors::Tensor;
use rand::Rng;
use tensors::TensorImpl;


struct XorGenerator {
    batch_size: usize
}

impl Iterator for XorGenerator {
    type Item = (TensorImpl<f64>, Vec<f64>);

    fn next(&mut self) -> Option<Self::Item> {
        Some(single_xor_batch(self.batch_size))
    }
    
}

impl XorGenerator {
    fn new(batch_size: usize) -> XorGenerator {
        XorGenerator { batch_size }
    }
}

fn single_xor_batch(batch_size: usize) -> (TensorImpl<f64>, Vec<f64>) {
    let x_size = vec![1, batch_size, 2];

    let known_xor = HashMap::from([
        (0 , ([0f64, 0f64], 0f64)),
        (1 , ([0f64, 1f64], 1f64)),
        (2 , ([1f64, 0f64], 1f64)),
        (3 , ([1f64, 1f64], 0f64)),
    ]);

    let mut y_result = Vec::<f64>::with_capacity(batch_size);
    let mut x_accumulator = Vec::<f64>::with_capacity(batch_size * 2);

    for _ in 0..batch_size {
        // Pick a random number between 0 and 3
        let choice = rand::thread_rng().gen_range(0..4);
        let (x, y) = known_xor.get(&choice).unwrap();

        y_result.push(y.clone());
        x_accumulator.extend(x.iter());
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
            let (x, y) = single_xor_batch(batch_size);

            assert_eq!(x.get_shape(), &vec![1, batch_size, 2]);
            assert_eq!(y.len(), batch_size);
        }
    }

    #[test]
    fn test_xor_generator() {
        let batch_size = 10;
        let mut generator = XorGenerator::new(batch_size);

        for _ in 0..10 {
            let (x, y) = generator.next().unwrap();

            assert_eq!(x.get_shape(), &vec![1, batch_size, 2]);
            assert_eq!(y.len(), batch_size);
        }
    }

}
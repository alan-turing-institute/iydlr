use interfaces::tensors::Tensor;
// use interfaces::tensors::TensorImpl;
use tensors::TensorImpl;

// pub const N: f64 = 10000.0;

pub fn pos_encoding(seq_len: usize, d: usize, n: usize) -> TensorImpl<f64> {
    if d % 2 != 0 {
        panic!("The dimension d must be even");
    }
    if d < seq_len {
        panic!("d must not be less than seq_len");
    }

    let mut accumulator = Vec::<f64>::with_capacity(d * d);

    let d_f64 = d as f64;
    let n = n as f64;

    // each row
    for k in 0..seq_len {
        //each pair of columns
        let k = k as f64;
        for i in 0..(d / 2) {
            let i = i as f64;
            accumulator.push((k / n.powf(2.0 * i / d_f64)).sin());
            accumulator.push((k / n.powf(2.0 * i / d_f64)).cos());
        }
    }

    println!("{:?}", accumulator);

    let shape = vec![d, d];
    TensorImpl::from_vec(&shape, &accumulator).unwrap()
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_pos_encoding() {
        // test the pos_encoding function
        let result = pos_encoding(4, 4, 100);
        // check that it returns a tensor of size 4x4
        assert_eq!(result.shape(), vec![4, 4]);

        let compare = [
            (result.at(vec![0, 0]).unwrap(), &0_f64),
            (result.at(vec![0, 1]).unwrap(), &1_f64),
            (result.at(vec![0, 2]).unwrap(), &0_f64),
            (result.at(vec![0, 3]).unwrap(), &1_f64),
            (result.at(vec![1, 0]).unwrap(), &0.8414709848078965_f64),
            (result.at(vec![1, 1]).unwrap(), &0.54030230586814_f64),
            (result.at(vec![1, 2]).unwrap(), &0.099833416646828_f64),
            (result.at(vec![1, 3]).unwrap(), &0.995004165278026_f64),
            (result.at(vec![2, 0]).unwrap(), &0.909297426825682_f64),
            (result.at(vec![2, 1]).unwrap(), &-0.416146836547142_f64),
            (result.at(vec![2, 2]).unwrap(), &0.198669330795061_f64),
            (result.at(vec![2, 3]).unwrap(), &0.980066577841242_f64),
        ];

        for (actual, expected) in compare {
            println!("{} {}", actual, expected);
            assert!((actual - expected).abs() < 10.0_f64.powf(-6.0));
        }
    }
}

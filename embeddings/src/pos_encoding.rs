use interfaces::{deep_learning::DLModule, tensors::{Element, RealElement, RealTensor, Tensor}};
// use interfaces::tensors::TensorImpl;
use tensors::TensorImpl;
use std::marker::PhantomData;


// Create PE layer with implements DLModule

pub struct PELayer<T: Tensor<E>, E: Element>{
    tensor_phantom: PhantomData<T>,
    tensor_element_phantom: PhantomData<E>,
}


impl<T, E> DLModule<T, E> for PELayer<T, E> 
where
    T: Tensor<E>,
    E: RealElement + Into<f64>,
{
    type DLModuleError = <T as Tensor<E>>::TensorError;

    fn forward(&self, x: &T) -> Result<T, Self::DLModuleError> {
        let input_shape = x.shape();
        let seq_len = input_shape[1];
        let d = input_shape[2];
        let pe: T = PELayer::pos_encoding(seq_len, d, 10000);
        
        Ok(x.clone().add(pe))
    }
    

    // This should return a empty vector
    fn params(&self) -> Vec<E> {
        Vec::<E>::new()
    }
    
}


impl<T, E> PELayer<T, E>
where
    T: Tensor<E>,
    E: Element + From<f64>,
{
    pub fn pos_encoding(seq_len: usize, d: usize, n: usize) -> T {
        if d % 2 != 0 {
            panic!("The dimension d must be even");
        }
        if d < 4 {
            panic!("d must be greater or equal to 4");
        }
    
        let mut accumulator = Vec::<E>::with_capacity(d * d);
    
        let d_f64 = d as f64;
        let n = n as f64;
    
        // each row
        for k in 0..seq_len {
            //each pair of columns
            let k = k as f64;
            for i in 0..(d / 2) {
                let i = i as f64;
                accumulator.push((k / n.powf(2.0 * i / d_f64)).sin().into());
                accumulator.push((k / n.powf(2.0 * i / d_f64)).cos().into());
            }
        }
    
        // Using PyTorch's behaviour as a reference
        let shape = vec![1, seq_len, d];

        T::from_vec(&shape, &accumulator).unwrap()
        // PELayer {
        //     x: T::from_vec(&shape, &accumulator).unwrap(),
        //     tensor_element_phantom: PhantomData
        // }
    }
}




#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_pos_encoding() {
        let result: TensorImpl<f64> = PELayer::pos_encoding(4, 4, 100);
        // check that it returns a tensor of size 4x4
        assert_eq!(result.shape(), vec![1, 4, 4]);

        let compare = [
            (result.at(vec![0, 0, 0]).unwrap(), &0_f64),
            (result.at(vec![0, 0, 1]).unwrap(), &1_f64),
            (result.at(vec![0, 0, 2]).unwrap(), &0_f64),
            (result.at(vec![0, 0, 3]).unwrap(), &1_f64),
            (result.at(vec![0, 1, 0]).unwrap(), &0.8414709848078965_f64),
            (result.at(vec![0, 1, 1]).unwrap(), &0.54030230586814_f64),
            (result.at(vec![0, 1, 2]).unwrap(), &0.099833416646828_f64),
            (result.at(vec![0, 1, 3]).unwrap(), &0.995004165278026_f64),
            (result.at(vec![0, 2, 0]).unwrap(), &0.909297426825682_f64),
            (result.at(vec![0, 2, 1]).unwrap(), &-0.416146836547142_f64),
            (result.at(vec![0, 2, 2]).unwrap(), &0.198669330795061_f64),
            (result.at(vec![0, 2, 3]).unwrap(), &0.980066577841242_f64),
        ];

        for (actual, expected) in compare {
            println!("{} {}", actual, expected);
            assert!((actual - expected).abs() < 10.0_f64.powf(-6.0));
        }
    }

    #[test]
    fn test_forward() {
        let original_data = (0..32).into_iter().map(|x| x as f64).collect::<Vec<f64>>();
        let target = PELayer{ tensor_phantom: PhantomData, tensor_element_phantom: PhantomData };
        let x = TensorImpl::from_vec(&vec![2, 4, 4], &original_data).unwrap();
   
        let result = target.forward(&x).unwrap();

        assert_eq!(result.shape(), vec![2, 4, 4]);

        assert_eq!(result.at(vec![0, 1, 0]).unwrap(), &4.8414709848078965_f64);
    }
}

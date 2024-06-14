use interfaces::{
    deep_learning::DLModule,
    tensors::{Element, Tensor},
};

pub struct Serial<T, E>
where
    T: Tensor<E>,
    E: Element,
{
    modules: Vec<Box<dyn DLModule<T, E, DLModuleError = <T as Tensor<E>>::TensorError>>>,
}

impl<T, E> DLModule<T, E> for Serial<T, E>
where
    T: Tensor<E>,
    E: Element,
{
    type DLModuleError = <T as Tensor<E>>::TensorError;

    fn forward(&self, x: &T) -> Result<T, Self::DLModuleError> {
        let mut tmp = x.clone();
        for module in self.modules.iter() {
            tmp = module.forward(&tmp)?
        }
        Ok(tmp.clone())
    }

    fn params(&self) -> Vec<E> {
        self.modules.iter().fold(Vec::new(), |mut acc, module| {
            acc.extend(module.params());
            acc
        })
    }
}

impl<T, E> Serial<T, E>
where
    T: Tensor<E>,
    E: Element,
{
    pub fn new(
        modules: Vec<Box<dyn DLModule<T, E, DLModuleError = <T as Tensor<E>>::TensorError>>>,
    ) -> Self {
        Serial { modules }
    }
}

#[cfg(test)]
mod tests {
    use crate::lin_layer::LinLayer;

    use super::*;
    use rand::seq;
    use tensors::TensorImpl;

    #[test]
    fn construct_serial() {
        let seed = 0;
        let _: Serial<TensorImpl<f64>, f64> = Serial::new(vec![
            Box::new(LinLayer::new(1, 3, seed)),
            Box::new(LinLayer::new(3, 1, seed)),
        ]);
    }

    #[test]
    fn forward_serial() {
        let seed = 0;
        let serial: Serial<TensorImpl<f64>, f64> = Serial::new(vec![
            Box::new(LinLayer::new(4, 3, seed)),
            Box::new(LinLayer::new(3, 5, seed)),
        ]);
        let x = TensorImpl::from_vec(&vec![3, 2, 4], &vec![4.0; 24]).unwrap();
        let out = serial.forward(&x).unwrap();
        assert_eq!(out.shape(), vec![3, 2, 5])
    }

    #[test]
    fn params_serial() {
        let seed = 0;
        let serial: Serial<TensorImpl<f64>, f64> = Serial::new(vec![
            Box::new(LinLayer::new(1, 3, seed)),
            Box::new(LinLayer::new(3, 1, seed)),
        ]);
        // w1 + b1 + w2 + b2
        assert_eq!(serial.params().len(), 3 + 3 + 3 + 1);
    }
}

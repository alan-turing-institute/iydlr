use crate::TensorImpl;
use interfaces::{
    tensors::{Element, RealElement, Tensor},
    utils::Pow,
};
use num_traits::identities::Zero;

impl From<f64> for TensorImpl<f64> {
    fn from(value: f64) -> Self {
        TensorImpl::from_vec(&vec![1, 1], &vec![value]).unwrap()
    }
}

impl Pow for TensorImpl<f64> {
    fn pow(self, exp: Self) -> Self {
        // exponent must be a single element tensor
        self.pow(exp.into_iter().next().unwrap())
    }
}

impl Element for TensorImpl<f64> {}

impl RealElement for TensorImpl<f64> {
    fn neg_inf() -> Self {
        TensorImpl::from(f64::neg_inf())
    }
}

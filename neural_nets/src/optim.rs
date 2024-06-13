use autodiff::node::Node;
use interfaces::tensors::{RealElement, RealTensor};

pub struct OptimSGD<T> {
    l_rate: f64,
    max_itr: usize,
    params: Vec<T>,
}

impl<T> OptimSGD<T> {
    pub fn new(l_rate: f64, max_itr: usize, params: Vec<T>) -> OptimSGD<T> {
        OptimSGD {
            l_rate,
            max_itr,
            params,
        }
    }
}

impl OptimSGD<Node<f64>> {
    pub fn zero_grad(&mut self) {
        for p in self.params.iter_mut() {
            p.set_grad(0.0)
        }
    }

    pub fn update(&mut self, itr: usize) {
        let mut l_rate = self.l_rate;
        if itr > self.max_itr.saturating_mul(3).saturating_div(4) {
            l_rate *= 0.1;
        }
        for p in self.params.iter_mut() {
            // println!("{:?}", p.grad());
            p.set_val(p.val() + (-l_rate * p.grad().unwrap()))
        }
    }
}

// fn bce<E>(y: E, y_pred: E) -> E
// where
//     E: RealElement + From<f64>,
// {
//     // -1 * [ y * (y_pred + 0.0001).ln()    +    (1 - y) * (1 - (y_pred - 0.0001)).ln() ]

//     E::from(-1.0)
//         * (y.clone() * (y_pred.clone() + E::from(0.0000001)).ln()
//             + (E::from(1.0) - y) * (E::from(1.0) - (y_pred - E::from(0.0000001))).ln())
// }

pub fn bce<T, E>(y: T, y_pred: T) -> T
where
    T: RealTensor<E>,
    E: RealElement + From<f64>,
{
    // -1 * [ y * (y_pred + 0.0001).ln()    +    (1 - y) * (1 - (y_pred - 0.0001)).ln() ]
    let t_ones = T::fill_with_clone(y.shape(), E::from(1.0));
    T::fill_with_clone(y.shape(), E::from(-1.0))
        * (y.clone() * (y_pred.clone() + E::from(0.0000001)).ln()
            + (t_ones.clone() + (y * E::from(-1.0)))
                * (t_ones + (y_pred + E::from(-0.0000001)) * E::from(-1.0)).ln())
}

/// Categorical (i.e. multi-label) cross entropy loss function.
pub fn cce<T, E>(y: T, y_pred: T) -> T
where
    T: RealTensor<E>,
    E: RealElement + From<f64>,
{
    let y_shape = y.shape();
    let t_ones = T::fill_with_clone(vec![y_shape[0], y_shape[2]], E::from(1.0));
    (y * y_pred.ln()).matmul(&t_ones.transpose()).unwrap() // TODO: minus 1 *
}

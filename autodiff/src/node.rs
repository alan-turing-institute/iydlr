use interfaces::tensors::{AsAnyhowError, RealTensor, Tensor};
use interfaces::{
    tensors::{Element, RealElement},
    utils::{Exp, Ln, Pow},
};
use num_traits::{Inv, Zero};
use std::any::Any;
use std::cell::{Ref, RefMut};
use std::error::Error;
use std::iter::zip;
use std::ops::MulAssign;
use std::{
    cell::RefCell,
    cmp::{Ordering, PartialOrd},
    fmt::Display,
    ops::{Add, AddAssign, Deref, DerefMut, Div, Mul, Sub},
    rc::Rc,
};
use tensors::TensorImpl;

type Ptr<N> = Rc<RefCell<N>>;

#[derive(Debug)]
pub enum NodeContent<T> {
    Sum(T, Option<T>, (Node<T>, Node<T>)),
    Prod(T, Option<T>, (Node<T>, Node<T>)),
    Quot(T, Option<T>, (Node<T>, Node<T>)),
    Exp(T, Option<T>, Node<T>),
    Ln(T, Option<T>, Node<T>),
    Pow(T, Option<T>, (Node<T>, Node<T>)),
    Matmul(T, Option<T>, (Node<T>, Node<T>)),
    Transpose(T, Option<T>, Node<T>),
    /// (value, gradient, parent, parent_shape)
    Reshape(T, Option<T>, Node<T>, Vec<usize>),
    /// (value, gradient, parent, dimension, parent_shape)
    DimSum(T, Option<T>, Node<T>, usize, Vec<usize>),
    Leaf(T, Option<T>),
}

/// A node in a computation graph.
#[derive(Debug)]
pub struct Node<T> {
    ptr: Ptr<NodeContent<T>>,
}

impl<T: RealTensor<f64> + From<f64>> Node<T> {
    pub fn new(val: T, grad: Option<T>) -> Self {
        Node {
            ptr: Rc::new(RefCell::new(NodeContent::new(val, grad))),
        }
    }

    pub fn val(&self) -> Ref<T> {
        Ref::map(self.ptr.borrow(), |node_content| node_content.val())
    }

    pub fn val_mut(&mut self) -> RefMut<T> {
        RefMut::map(self.ptr.borrow_mut(), |node_content| node_content.val_mut())
    }

    pub fn grad(&self) -> Ref<Option<T>> {
        Ref::map(self.ptr.borrow(), |node_content| node_content.grad())
    }

    pub fn set_grad(&mut self, new_grad: T) {
        self.ptr.borrow_mut().set_grad(new_grad)
    }

    pub fn set_val(&mut self, new_val: T) {
        self.ptr.borrow_mut().set_val(new_val)
    }

    pub fn add_assign_grad(&mut self, mut new_grad: T) {
        if let Some(grad) = self.grad().as_ref() {
            new_grad += grad
        };
        self.set_grad(new_grad)
    }

    // Set the gradient and initiate backward propagation.
    // Propagate a given gradient on the `grad` of each associated Node.
    // Assumes the `grad` on self is not None.
    pub fn backward(&mut self, grad: T) {
        let self_val = self.val().clone();
        // println!("backward: {:?}", self.shape());

        match *self.ptr.borrow_mut() {
            NodeContent::Sum(_, _, (ref mut np1, ref mut np2)) => {
                // println!("SUM");
                let grad_shape = grad.shape();
                let np1_shape = np1.shape();
                let np2_shape = np2.shape();

                if np1_shape != grad_shape {
                    for (dim_idx, (grad_dim, np1_dim)) in
                        zip(grad_shape.iter(), np1_shape.iter()).enumerate()
                    {
                        if grad_dim != np1_dim {
                            assert!(
                                grad_dim > np1_dim,
                                "broadcasting should only increase the dimensionality"
                            );
                            np1.backward(grad.dim_sum(vec![dim_idx]));
                        }
                    }
                } else {
                    np1.backward(grad.clone());
                }
                if np2_shape != grad_shape {
                    for (dim_idx, (grad_dim, np2_dim)) in zip(grad_shape, np2_shape).enumerate() {
                        if grad_dim != np2_dim {
                            assert!(
                                grad_dim > np2_dim,
                                "broadcasting should only increase the dimensionality"
                            );
                            np2.backward(grad.dim_sum(vec![dim_idx]));
                        }
                    }
                } else {
                    np2.backward(grad.clone());
                }
            }
            NodeContent::Prod(_, _, (ref mut np1, ref mut np2)) => {
                // println!("PROD");
                let np1_grad = grad.clone() * np2.val().deref();
                let np2_grad = grad.clone() * np1.val().deref();
                np1.backward(np1_grad);
                np2.backward(np2_grad);
            }
            NodeContent::Quot(_, _, (ref mut np_num, ref mut np_denom)) => {
                let minus_one = <f64 as Into<T>>::into(-1_f64);

                let np_num_grad = grad.clone() / np_denom.val().to_owned();

                let np_denom_grad = grad.clone() * -1_f64 * np_num.val().to_owned()
                    / np_denom.val().to_owned().pow(2_f64);

                np_num.backward(np_num_grad);
                np_denom.backward(np_denom_grad);
            }
            NodeContent::Exp(_, _, ref mut np) => {
                // println!("EXP");
                let np_grad = grad.clone() * self_val;
                np.backward(np_grad);
            }
            NodeContent::Ln(_, _, ref mut np) => {
                // println!("LN");
                let np_grad =
                    (T::fill_with_clone(np.shape(), 1.0) / np.val().clone()) * grad.clone();
                np.backward(np_grad);
            }
            NodeContent::Pow(_, _, (ref mut np_b, ref mut np_e)) => {
                // println!("POW");
                // exponent . base^(exponent - 1)
                let b_val = np_b.val().clone();
                let e_val: f64 = np_e.val().clone().into_iter().next().unwrap();

                let np_b_grad = b_val.clone().pow(e_val - 1.0) * grad.clone() * e_val;
                let np_e_grad = b_val.clone().pow(e_val.to_owned()) * b_val.ln() * grad.clone();

                // base^exponent . ln(base)
                np_b.backward(np_b_grad);
                np_e.backward(np_e_grad);
            }
            NodeContent::Matmul(_, _, (ref mut np1, ref mut np2)) => {
                // println!("MATMUL");
                // println!("grad: {:?}", grad);
                // println!("grad shape: {:?}", grad.shape());
                // println!("np1.transpose shape: {:?}", np1.transpose().shape());
                // println!("np2 shape: {:?}", np2.shape());
                let np1_grad = grad.clone().matmul(&np2.val().deref().transpose()).unwrap();
                let np2_grad = np1.val().deref().transpose().matmul(&grad).unwrap();
                np1.backward(np1_grad);
                np2.backward(np2_grad);
            }
            NodeContent::Transpose(_, _, ref mut np1) => {
                // println!("TRANSPOSE");
                np1.backward(grad.transpose());
            }

            NodeContent::Reshape(_, _, ref mut np1, ref mut old_shape) => {
                // println!("RESHAPE");
                // println!("old_shape: {:?}", old_shape);
                // println!("grad_shape: {:?}", grad.shape());
                let mut reshaped_grad = grad.clone();
                reshaped_grad.reshape(old_shape.clone());
                np1.backward(reshaped_grad)
            }
            NodeContent::DimSum(_, _, ref mut np1, ref mut dim, ref mut old_shape) => {
                // println!("DIMSUM");
                let reshaped_grad = grad.clone();
                for _ in 0..old_shape[*dim] {
                    reshaped_grad.concat(&reshaped_grad, *dim);
                }
                np1.backward(reshaped_grad)
            }
            NodeContent::Leaf(_, _) => {} // Do nothing.
        }
        // println!("self shape: {:?}", self.shape());
        // println!("grad shape: {:?}", grad.shape());
        self.add_assign_grad(grad);
    }
}

impl<T> From<NodeContent<T>> for Node<T> {
    fn from(value: NodeContent<T>) -> Self {
        Node {
            ptr: Rc::new(RefCell::new(value)),
        }
    }
}

impl<T: RealTensor<f64> + From<f64>> NodeContent<T> {
    pub fn new(val: T, grad: Option<T>) -> Self {
        NodeContent::Leaf(val, grad)
    }

    pub fn val(&self) -> &T {
        match self {
            NodeContent::Sum(val, _, _)
            | NodeContent::Prod(val, _, _)
            | NodeContent::Quot(val, _, _)
            | NodeContent::Exp(val, _, _)
            | NodeContent::Ln(val, _, _)
            | NodeContent::Pow(val, _, _)
            | NodeContent::Matmul(val, _, _)
            | NodeContent::Transpose(val, _, _)
            | NodeContent::Reshape(val, _, _, _)
            | NodeContent::DimSum(val, _, _, _, _)
            | NodeContent::Leaf(val, _) => val,
        }
    }

    pub fn val_mut(&mut self) -> &mut T {
        match self {
            NodeContent::Sum(val, _, _)
            | NodeContent::Prod(val, _, _)
            | NodeContent::Quot(val, _, _)
            | NodeContent::Exp(val, _, _)
            | NodeContent::Ln(val, _, _)
            | NodeContent::Pow(val, _, _)
            | NodeContent::Matmul(val, _, _)
            | NodeContent::Transpose(val, _, _)
            | NodeContent::Reshape(val, _, _, _)
            | NodeContent::DimSum(val, _, _, _, _)
            | NodeContent::Leaf(val, _) => val,
        }
    }

    pub fn grad(&self) -> &Option<T> {
        match self {
            NodeContent::Sum(_, grad, _)
            | NodeContent::Prod(_, grad, _)
            | NodeContent::Quot(_, grad, _)
            | NodeContent::Exp(_, grad, _)
            | NodeContent::Ln(_, grad, _)
            | NodeContent::Pow(_, grad, _)
            | NodeContent::Matmul(_, grad, _)
            | NodeContent::Transpose(_, grad, _)
            | NodeContent::Reshape(_, grad, _, _)
            | NodeContent::DimSum(_, grad, _, _, _)
            | NodeContent::Leaf(_, grad) => grad,
        }
    }

    pub fn set_grad(&mut self, new_grad: T) {
        let g: &mut Option<T> = match self {
            NodeContent::Sum(_, grad, _)
            | NodeContent::Prod(_, grad, _)
            | NodeContent::Quot(_, grad, _)
            | NodeContent::Exp(_, grad, _)
            | NodeContent::Ln(_, grad, _)
            | NodeContent::Pow(_, grad, _)
            | NodeContent::Matmul(_, grad, _)
            | NodeContent::Transpose(_, grad, _)
            | NodeContent::Reshape(_, grad, _, _)
            | NodeContent::DimSum(_, grad, _, _, _)
            | NodeContent::Leaf(_, grad) => grad,
        };

        *g = Some(new_grad);
    }

    pub fn set_val(&mut self, new_val: T) {
        let v: &mut T = match self {
            NodeContent::Sum(val, _, _)
            | NodeContent::Prod(val, _, _)
            | NodeContent::Quot(val, _, _)
            | NodeContent::Exp(val, _, _)
            | NodeContent::Ln(val, _, _)
            | NodeContent::Pow(val, _, _)
            | NodeContent::Matmul(val, _, _)
            | NodeContent::Transpose(val, _, _)
            | NodeContent::Reshape(val, _, _, _)
            | NodeContent::DimSum(val, _, _, _, _)
            | NodeContent::Leaf(val, _) => val,
        };

        *v = new_val;
    }

    pub fn add_assign_grad(&mut self, new_grad: T) {
        let g: &mut Option<T> = match self {
            NodeContent::Sum(_, grad, _)
            | NodeContent::Prod(_, grad, _)
            | NodeContent::Quot(_, grad, _)
            | NodeContent::Exp(_, grad, _)
            | NodeContent::Ln(_, grad, _)
            | NodeContent::Pow(_, grad, _)
            | NodeContent::Matmul(_, grad, _)
            | NodeContent::Transpose(_, grad, _)
            | NodeContent::Reshape(_, grad, _, _)
            | NodeContent::DimSum(_, grad, _, _, _)
            | NodeContent::Leaf(_, grad) => grad,
        };

        match g {
            Some(grad) => *g = Some((*grad).clone() + new_grad),
            None => *g = Some(new_grad),
        }
    }
}

impl<T: RealTensor<f64> + From<f64>> Add<Node<T>> for Node<T> {
    type Output = Node<T>;

    fn add(self, rhs: Node<T>) -> Self::Output {
        // self and rhs are Rc pointers, so the clone has minimal overhead
        NodeContent::Sum(
            self.clone().val().clone() + rhs.clone().val().deref(),
            None,
            (self, rhs),
        )
        .into()
    }
}

impl<T: RealTensor<f64> + From<f64>> Add<&Node<T>> for Node<T> {
    type Output = Node<T>;

    fn add(self, rhs: &Node<T>) -> Self::Output {
        // self and rhs are Rc pointers, so the clone has minimal overhead
        NodeContent::Sum(
            self.clone().val().clone() + rhs.clone().val().deref(),
            None,
            (self, rhs.clone()),
        )
        .into()
    }
}

impl<T: RealTensor<f64> + From<f64>> Add<f64> for Node<T> {
    type Output = Node<T>;

    fn add(self, rhs: f64) -> Self::Output {
        // self and rhs are Rc pointers, so the clone has minimal overhead
        NodeContent::Sum(self.clone().val().clone() + rhs, None, (self, rhs.into())).into()
    }
}

impl<T: RealTensor<f64> + From<f64>> Sub<Node<T>> for Node<T> {
    type Output = Node<T>;

    fn sub(self, mut rhs: Node<T>) -> Self::Output {
        rhs = rhs * Node::from(-1.0);
        NodeContent::Sum(
            self.clone().val().clone() + rhs.clone().val().deref(),
            None,
            (self, rhs),
        )
        .into()
    }
}

impl<T: RealTensor<f64> + From<f64>> Sub<&Node<T>> for Node<T> {
    type Output = Node<T>;

    fn sub(self, rhs: &Node<T>) -> Self::Output {
        let rhs_neg = Node::from(-1.0) * rhs;
        NodeContent::Sum(
            self.clone().val().clone() + rhs_neg.clone().val().deref(),
            None,
            (self, rhs_neg),
        )
        .into()
    }
}

impl<T: RealTensor<f64> + From<f64>> Sub<f64> for Node<T> {
    type Output = Node<T>;

    fn sub(self, rhs: f64) -> Self::Output {
        let rhs_neg: Node<T> = Node::from(-1.0 * rhs);
        NodeContent::Sum(
            self.clone().val().clone() + rhs_neg.clone().val().deref(),
            None,
            (self, rhs_neg),
        )
        .into()
    }
}

impl<T: RealTensor<f64> + From<f64>> Mul<Node<T>> for Node<T> {
    type Output = Node<T>;

    fn mul(self, rhs: Node<T>) -> Self::Output {
        NodeContent::Prod(
            self.clone().val().clone() * rhs.clone().val().deref(),
            None,
            (self, rhs),
        )
        .into()
    }
}

impl<T: RealTensor<f64> + From<f64>> Mul<&Node<T>> for Node<T> {
    type Output = Node<T>;

    fn mul(self, rhs: &Node<T>) -> Self::Output {
        NodeContent::Prod(
            self.clone().val().clone() * rhs.clone().val().deref(),
            None,
            (self, rhs.clone()),
        )
        .into()
    }
}

impl<T: RealTensor<f64> + From<f64>> MulAssign<Node<T>> for Node<T> {
    fn mul_assign(&mut self, rhs: Node<T>) {
        *self = NodeContent::Prod(
            self.clone().val().clone() * rhs.clone().val().deref(),
            None,
            (self.clone(), rhs),
        )
        .into()
    }
}

impl<T: RealTensor<f64> + From<f64>> MulAssign<&Node<T>> for Node<T> {
    fn mul_assign(&mut self, rhs: &Node<T>) {
        *self = NodeContent::Prod(
            self.clone().val().clone() * rhs.clone().val().deref(),
            None,
            (self.clone(), rhs.clone()),
        )
        .into()
    }
}

impl<T: RealTensor<f64> + From<f64>> Mul<f64> for Node<T> {
    type Output = Node<T>;

    fn mul(self, rhs: f64) -> Self::Output {
        NodeContent::Prod(self.clone().val().clone() * rhs, None, (self, rhs.into())).into()
    }
}

impl<T: RealTensor<f64> + From<f64>> MulAssign<f64> for Node<T> {
    fn mul_assign(&mut self, rhs: f64) {
        *self = NodeContent::Prod(
            self.clone().val().clone() * rhs,
            None,
            (self.clone(), rhs.into()),
        )
        .into()
    }
}

impl<T: RealTensor<f64> + From<f64>> Div<Node<T>> for Node<T> {
    type Output = Node<T>;

    fn div(self, rhs: Node<T>) -> Self::Output {
        NodeContent::Quot(
            self.clone().val().clone() / rhs.clone().val().deref(),
            None,
            (self, rhs),
        )
        .into()
    }
}

impl<T: RealTensor<f64> + From<f64>> Div<&Node<T>> for Node<T> {
    type Output = Node<T>;

    fn div(self, rhs: &Node<T>) -> Self::Output {
        NodeContent::Quot(
            self.clone().val().clone() / rhs.clone().val().deref(),
            None,
            (self, rhs.clone()),
        )
        .into()
    }
}

impl<T: RealTensor<f64> + From<f64>> Div<f64> for Node<T> {
    type Output = Node<T>;

    fn div(self, rhs: f64) -> Self::Output {
        NodeContent::Quot(self.clone().val().clone() / rhs, None, (self, rhs.into())).into()
    }
}

impl<T: RealTensor<f64> + From<f64>> Exp for Node<T> {
    fn exp(self) -> Self {
        NodeContent::Exp(self.clone().val().clone().exp(), None, self).into()
    }
}

impl<T: RealTensor<f64> + From<f64>> Ln for Node<T> {
    fn ln(self) -> Self {
        NodeContent::Ln(self.clone().val().clone().ln(), None, self).into()
    }
}

impl<T: RealTensor<f64> + From<f64>> Pow<f64> for Node<T> {
    fn pow(self, exp: f64) -> Self {
        NodeContent::Pow(
            self.clone().val().clone().pow(exp),
            None,
            (self, exp.into()),
        )
        .into()
    }
}

impl<T: RealTensor<f64> + From<f64>> AddAssign for Node<T> {
    fn add_assign(&mut self, _rhs: Self) {
        *self = self.clone() + _rhs;
    }
}

impl<T: RealTensor<f64> + From<f64>> AddAssign<&Node<T>> for Node<T> {
    fn add_assign(&mut self, _rhs: &Self) {
        *self = self.clone() + _rhs;
    }
}

impl<T: RealTensor<f64> + From<f64>> Display for NodeContent<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node: {:?}", self)
    }
}

impl<T: RealTensor<f64> + From<f64>> Display for Node<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node: {:?}", self.ptr.deref().borrow())
    }
}

impl<T: RealTensor<f64> + From<f64>> Clone for Node<T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr.clone(),
        }
    }
}

impl<T: RealTensor<f64> + From<f64>> From<f64> for Node<T> {
    fn from(value: f64) -> Self {
        Node::new(value.into(), None)
    }
}

impl<T: RealTensor<f64> + From<f64>> Zero for Node<T> {
    fn zero() -> Self {
        Node::new(T::zero(), None)
    }

    fn is_zero(&self) -> bool {
        self.val().is_zero()
    }

    fn set_zero(&mut self) {
        self.val_mut().set_zero()
    }
}

impl<T: RealTensor<f64> + From<f64>> IntoIterator for Node<T> {
    type Item = f64;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> std::vec::IntoIter<Self::Item> {
        self.val().clone().into_iter()
    }
}

impl<T: RealTensor<f64> + From<f64>> Into<Vec<f64>> for Node<T> {
    fn into(self) -> Vec<f64> {
        self.into_iter().collect()
    }
}

impl From<TensorImpl<f64>> for Node<TensorImpl<f64>> {
    fn from(value: TensorImpl<f64>) -> Self {
        Node::new(value, None)
    }
}

impl<T> Tensor<f64> for Node<T>
where
    T: RealTensor<f64> + From<f64>,
{
    type TensorError = <T as Tensor<f64>>::TensorError;

    fn at(&self, idxs: Vec<usize>) -> Option<&f64> {
        todo!()
    }
    fn at_mut(&mut self, idxs: Vec<usize>) -> Option<&mut f64> {
        todo!()
    }
    fn concat(&self, other: &Self, dim: usize) -> Result<Self, Self::TensorError> {
        todo!()
    }
    fn dim_sum(&self, dims: Vec<usize>) -> Self {
        if dims.len() != 1 {
            todo!()
        }
        NodeContent::DimSum(
            self.val().dim_sum(dims.clone()),
            None,
            self.clone(),
            dims[0],
            self.shape(),
        )
        .into()
    }
    fn fill_with_clone(shape: Vec<usize>, element: f64) -> Self {
        Node::new(T::fill_with_clone(shape, element), None)
    }
    fn from_vec(shape: &Vec<usize>, data: &Vec<f64>) -> Result<Self, Self::TensorError> {
        Ok(Node::new(T::from_vec(shape, data)?, None))
    }
    fn matmul(&self, other: &Self) -> Result<Self, Self::TensorError> {
        let value = self.val().matmul(other.val().deref())?;
        Ok(NodeContent::Matmul(value, None, (self.clone(), other.clone())).into())
    }
    fn reshape(&mut self, new_shape: Vec<usize>) {
        let old_shape = self.shape();
        let mut new_tensor = self.val().clone();
        new_tensor.reshape(new_shape);
        *self = NodeContent::Reshape(new_tensor, None, self.clone(), old_shape).into()
    }
    fn shape(&self) -> Vec<usize> {
        self.val().shape()
    }
    fn transpose(&self) -> Self {
        NodeContent::Transpose(self.val().transpose(), None, self.clone()).into()
    }
}

impl<T> RealTensor<f64> for Node<T>
where
    T: RealTensor<f64> + From<f64>,
{
    fn fill_from_f64(shape: Vec<usize>, data: f64) -> Self {
        todo!()
    }

    fn softmax(&self, dim: usize) -> Self {
        let t_small = f64::EPSILON;
        let data_exp = self.clone().exp();
        let sum = data_exp.dim_sum(vec![dim]);
        data_exp / (sum + t_small)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    // #[test]
    // fn test_new() {
    //     let node = NodeContent::<f64>::new(3.1, Some(0.4));
    //     assert_eq!(node.val(), &3.1_f64);
    //     assert_eq!(node.grad(), &Some(0.4));
    // }

    // #[test]
    // fn test_set_grad() {
    //     let mut node = NodeContent::<f64>::new(3.1, None);
    //     assert_eq!(node.val(), &3.1_f64);
    //     assert_eq!(node.grad(), &None);

    //     node.set_grad(0.4);

    //     assert_eq!(node.val(), &3.1_f64);
    //     assert_eq!(node.grad(), &Some(0.4));

    //     node.set_grad(0.7);

    //     assert_eq!(node.val(), &3.1_f64);
    //     assert_eq!(node.grad(), &Some(0.7));
    // }

    // #[test]
    // fn test_set_grad_node_ptr() {
    //     let mut node = Node::new(3.1, None);
    //     assert_eq!(node.val(), 3.1_f64);
    //     assert_eq!(node.grad(), None);

    //     let node_clone = node.clone();
    //     assert_eq!(node_clone.val(), 3.1_f64);
    //     assert_eq!(node_clone.grad(), None);

    //     node.set_grad(0.4);

    //     assert_eq!(node.val(), 3.1_f64);
    //     assert_eq!(node.grad(), Some(0.4));

    //     assert_eq!(node_clone.val(), 3.1_f64);
    //     assert_eq!(node_clone.grad(), Some(0.4));

    //     node.set_grad(0.7);

    //     assert_eq!(node.val(), 3.1_f64);
    //     assert_eq!(node.grad(), Some(0.7));

    //     assert_eq!(node_clone.val(), 3.1_f64);
    //     assert_eq!(node_clone.grad(), Some(0.7));
    // }

    // #[test]
    // fn test_add_assign_grad() {
    //     let mut node = NodeContent::<f64>::new(3.1, None);
    //     assert_eq!(node.val(), &3.1_f64);
    //     assert_eq!(node.grad(), &None);

    //     node.add_assign_grad(0.4);

    //     assert_eq!(node.val(), &3.1_f64);
    //     assert_eq!(node.grad(), &Some(0.4));

    //     node.add_assign_grad(0.7);

    //     assert_eq!(node.val(), &3.1_f64);
    //     assert_eq!(node.grad(), &Some(1.1));
    // }

    // #[test]
    // fn test_add_assign_grad_node_ptr() {
    //     let mut node = Node::new(3.1, None);

    //     assert_eq!(node.val(), 3.1_f64);
    //     assert_eq!(node.grad(), None);

    //     node.add_assign_grad(0.4);

    //     assert_eq!(node.val(), 3.1_f64);
    //     assert_eq!(node.grad(), Some(0.4));

    //     node.add_assign_grad(0.7);

    //     assert_eq!(node.val(), 3.1_f64);
    //     assert_eq!(node.grad(), Some(1.1));
    // }

    // #[test]
    // fn test_add() {
    //     let node1 = NodeContent::<f64>::new(3.1, Some(0.4));
    //     let node2 = NodeContent::<f64>::new(22.2, None);

    //     let result = node1 + node2;
    //     assert_eq!(result.val(), &25.3_f64);
    //     assert_eq!(result.grad(), &None);
    // }

    // #[test]
    // fn test_add_node() {
    //     let node1 = Node::<f64>::new(3.1, Some(0.4));
    //     let node2 = Node::<f64>::new(22.2, None);

    //     let result = node1 + node2;
    //     assert_eq!(result.val(), 25.3_f64);
    //     assert_eq!(result.grad(), None);
    // }

    // #[test]
    // fn test_add_assign_node() {
    //     let mut node1 = Node::<f64>::new(3.1, Some(0.4));
    //     let node2 = Node::<f64>::new(22.2, None);

    //     node1 += node2;
    //     assert_eq!(node1.val(), 25.3_f64);
    //     assert_eq!(node1.grad(), None);
    // }

    // #[test]
    // fn test_sub_node() {
    //     let node1 = Node::<f64>::new(5.0, Some(0.4));
    //     let node2 = Node::<f64>::new(7.0, None);

    //     let result = node1 - node2;
    //     assert_eq!(result.val(), -2.0_f64);
    //     assert_eq!(result.grad(), None);
    // }

    // #[test]
    // fn test_mul() {
    //     let node1 = NodeContent::<f64>::new(3.1, Some(0.4));
    //     let node2 = NodeContent::<f64>::new(22.2, None);

    //     let result = node1 * node2;
    //     assert_eq!(result.val(), &68.82_f64);
    //     assert_eq!(result.grad(), &None);
    // }

    // #[test]
    // fn test_mul_node_ptr() {
    //     let node1 = NodeContent::<f64>::new(3.1, Some(0.4));
    //     let node2 = NodeContent::<f64>::new(22.2, None);

    //     let np1: Node<f64> = node1.into();
    //     let np2 = node2.into();

    //     let result = np1 * np2;
    //     assert_eq!(result.val(), 68.82_f64);
    //     assert_eq!(result.grad(), None);
    // }

    // #[test]
    // fn test_div() {
    //     let node1 = NodeContent::<f64>::new(3.1, Some(0.4));
    //     let node2 = NodeContent::<f64>::new(22.2, None);

    //     let result = node1 / node2;
    //     assert_eq!(result.val(), &0.13963963963963966_f64);
    //     assert_eq!(result.grad(), &None);
    // }

    // #[test]
    // fn test_div_node_ptr() {
    //     let node1 = NodeContent::<f64>::new(3.1, Some(0.4));
    //     let node2 = NodeContent::<f64>::new(22.2, None);

    //     let np1: Node<f64> = node1.into();
    //     let np2 = node2.into();

    //     let result = np1 / np2;
    //     assert_eq!(result.val(), 0.13963963963963966_f64);
    //     assert_eq!(result.grad(), None);
    // }

    // #[test]
    // fn test_div_by_zero() {
    //     let node1 = NodeContent::<f64>::new(3.1, Some(0.4));
    //     let node2 = NodeContent::<f64>::new(0.0, None);

    //     let result = node1 / node2;
    //     assert_eq!(result.val(), &f64::INFINITY);
    // }

    // #[test]
    // fn test_pow() {
    //     let node1 = NodeContent::<f64>::new(3.1, Some(0.4));
    //     let node2 = NodeContent::<f64>::new(22.2, None);

    //     let result = node1.pow(node2);
    //     assert_eq!(result.val(), &80952376567.60643_f64);
    //     assert_eq!(result.grad(), &None);
    // }

    // #[test]
    // fn test_pow_node_ptr() {
    //     let node1 = NodeContent::<f64>::new(3.1, Some(0.4));
    //     let node2 = NodeContent::<f64>::new(22.2, None);

    //     let np1: Node<f64> = node1.into();
    //     let np2 = node2.into();

    //     let result = np1.pow(np2);
    //     assert_eq!(result.val(), 80952376567.60643_f64);
    //     assert_eq!(result.grad(), None);
    // }
    // // #[test]
    // // fn test_fmt() {
    // //     let node = Node::<f64>::new(3.1, None);
    // //     println!("{}", node);
    // // }

    // #[test]
    // fn test_backward_on_sum() {
    //     let node1 = Node::new(1.1, None);
    //     let node2 = Node::new(2.2, None);

    //     let mut node = node1.clone() + node2.clone();

    //     assert!(node.grad().is_none());
    //     assert!(node1.grad().is_none());
    //     assert!(node2.grad().is_none());

    //     node.backward(5.0);

    //     assert!(node.grad().is_some());
    //     assert_eq!(node.grad().unwrap(), 5.0_f64);
    //     assert!(node1.grad().is_some());
    //     assert_eq!(node1.grad().unwrap(), 5.0_f64);
    //     assert!(node2.grad().is_some());
    //     assert_eq!(node2.grad().unwrap(), 5.0_f64);
    // }

    // #[test]
    // fn test_backward_on_prod() {
    //     let node1 = Node::new(1.1, None);
    //     let node2 = Node::new(2.2, None);

    //     let mut node = node1.clone() * node2.clone();

    //     assert!(node.grad().is_none());
    //     assert!(node1.grad().is_none());
    //     assert!(node2.grad().is_none());

    //     node.backward(5.0);

    //     assert!(node.grad().is_some());
    //     assert_eq!(node.grad().unwrap(), 5.0_f64);

    //     assert!(node1.grad().is_some());
    //     assert_eq!(node1.grad().unwrap(), 11.0_f64);
    //     assert!(node2.grad().is_some());
    //     assert_eq!(node2.grad().unwrap(), 5.5_f64);
    // }

    // #[test]
    // fn test_backward_on_prod_sum() {
    //     let node_a = Node::new(3.0, None);
    //     let node_b = Node::new(2.0, None);
    //     let node_c = Node::new(2.0, None);

    //     let node_d = node_a.clone() + node_b.clone();
    //     let mut node_f = node_d.clone() * node_c.clone();

    //     // Check all grads are None initially.
    //     assert!(node_f.grad().is_none());
    //     assert!(node_a.grad().is_none());
    //     assert!(node_b.grad().is_none());
    //     assert!(node_c.grad().is_none());
    //     assert!(node_d.grad().is_none());

    //     node_f.backward(10.0);

    //     // Check all grads have been populated.
    //     assert!(node_f.grad().is_some());
    //     assert_eq!(node_f.grad().unwrap(), 10.0_f64);

    //     assert!(node_d.grad().is_some());
    //     assert_eq!(node_d.grad().unwrap(), 20.0_f64);
    //     assert!(node_c.grad().is_some());
    //     assert_eq!(node_c.grad().unwrap(), 50.0_f64);

    //     assert!(node_a.grad().is_some());
    //     assert_eq!(node_a.grad().unwrap(), 20.0_f64);
    //     assert!(node_b.grad().is_some());
    //     assert_eq!(node_b.grad().unwrap(), 20.0_f64);
    // }

    // #[test]
    // fn test_backward_on_2x_squared_plus_exp_5x() {
    //     // Expression: f(x) = 2x^2 + exp(5x)

    //     let node_x = Node::new(3.0, None);

    //     let node_2 = Node::new(2.0, None);
    //     let node_2_ = Node::new(2.0, None);
    //     let node_5 = Node::new(5.0, None);

    //     let node_5x = node_5.clone() * node_x.clone();
    //     let node_exp_5x = node_5x.clone().exp();
    //     let node_x_squared = node_x.clone().pow(node_2.clone());
    //     let node_2x_squared = node_x_squared.clone() * node_2_.clone();

    //     let mut node_f = node_exp_5x.clone() + node_2x_squared.clone();

    //     node_f.backward(1.0);

    //     // Check all grads have been populated.
    //     assert_eq!(node_f.grad().unwrap(), 1.0_f64);

    //     // w.r.t. the exp(5x) node the grad is 1.
    //     assert_eq!(node_exp_5x.grad().unwrap(), 1.0_f64);

    //     // w.r.t. the 2x^2 node the grad is 1.
    //     assert_eq!(node_2x_squared.grad().unwrap(), 1.0_f64);

    //     // w.r.t. the x^2 node the grad is 2.
    //     assert_eq!(node_x_squared.grad().unwrap(), 2.0_f64);

    //     // w.r.t. the 5x node the grad is exp(5*3)
    //     assert_eq!(node_5x.grad().unwrap(), 3269017.372472110639302_f64);

    //     // w.r.t. the 2 node that is the exponent of x^2, the grad is 2 * 3^2 * ln(3) = 19.775021196025975.
    //     assert_eq!(node_2.grad().unwrap(), 19.775021196025975_f64);

    //     // w.r.t. the 2 node that multiplies the x^2, the grad is 3^2:
    //     assert_eq!(node_2_.grad().unwrap(), 9.0_f64);

    //     // w.r.t. the 5 node that multiplies the x, the grad is 3 * e^(5*3) =
    //     assert_eq!(node_5.grad().unwrap(), 9807052.117416331917906_f64);

    //     // df/dx = 4x + 5 exp(5x)
    //     // So, w.r.t. the x node, the grad is df/dx(3) = 12 + 5 * exp(15) = 16345098.862360553196509
    //     assert_eq!(node_x.grad().unwrap(), 16345098.862360553196509_f64);
    // }

    #[test]
    fn test_overly_complicated_identity_function() {
        // Expression: f(x) = exp(
        //  x*(
        //    ln(
        //     (x+3.14)
        //    )/x
        //   )
        // ) - 3.14 = x
        let node_x = Node::new(TensorImpl::fill_with_clone(vec![2, 2], 3.0), None);
        let node_314 = Node::new(TensorImpl::fill_with_clone(vec![2, 2], 3.14), None);
        let node_1 = node_x.clone() + node_314.clone();
        let node_1 = node_1.clone().ln();
        let node_1 = node_1.clone() / node_x.clone();
        let node_1 = node_1.clone() * node_x.clone();
        let node_1 = node_1.clone().exp();
        let mut node_1 = node_1.clone() + (Node::from(-1.0) * node_314.clone());
        node_1.backward(TensorImpl::fill_with_clone(vec![2, 2], 1.0));
        assert!(f64::abs((node_x.grad().clone().unwrap() - 1.0_f64).get_data()[0]) < 1e-10);
    }

    // #[test]
    // fn test_powers() {
    //     let value = 3.5;
    //     let node_x_1 = Node::new(value.clone(), None);
    //     let node_xx = node_x_1.clone() * node_x_1.clone();
    //     let mut node_xxx = node_xx.clone() * node_x_1.clone();
    //     node_xxx.backward(1.0);
    //     let grad1 = node_x_1.grad().unwrap();

    //     let node_x_2 = Node::new(value.clone(), None);
    //     let node_3 = Node::new(3.0, None);
    //     let mut node_x_cubed = node_x_2.clone().pow(node_3.clone());
    //     node_x_cubed.backward(1.0);
    //     let grad2 = node_x_2.grad().unwrap();
    //     assert_eq!(grad1, grad2);
    //     assert_eq!(grad1, 3.0 * value.clone().pow(2.0));
    // }

    #[test]
    fn test_cyclic_graph() {
        let val_a = TensorImpl::fill_with_clone(vec![2, 3], 1.1);
        let val_b = TensorImpl::fill_with_clone(vec![2, 3], 1.2);
        let val_c = TensorImpl::fill_with_clone(vec![2, 3], 1.3);
        let val_d = TensorImpl::fill_with_clone(vec![2, 3], 1.4);
        let a1 = Node::new(val_a.clone(), None);
        let b1 = Node::new(val_b.clone(), None);
        let c1 = Node::new(val_c.clone(), None);
        let d1 = Node::new(val_d.clone(), None);
        let ab1 = a1.clone() + b1.clone();
        let abc1 = ab1.clone() + c1.clone();
        let abd1 = ab1.clone() + d1.clone();
        let mut result1 = abc1.clone() + abd1.clone();
        result1.backward(TensorImpl::fill_with_clone(vec![2, 3], 1.0));
        let grad_a1 = a1.grad().clone().unwrap();
        let grad_b1 = b1.grad().clone().unwrap();
        let grad_c1 = c1.grad().clone().unwrap();
        let grad_d1 = d1.grad().clone().unwrap();

        let a2 = Node::new(val_a.clone(), None);
        let b2 = Node::new(val_b.clone(), None);
        let c2 = Node::new(val_c.clone(), None);
        let d2 = Node::new(val_d.clone(), None);
        let mut result2 =
            a2.clone() + a2.clone() + b2.clone() + b2.clone() + c2.clone() + d2.clone();
        result2.backward(TensorImpl::fill_with_clone(vec![2, 3], 1.0));
        let grad_a2 = a2.grad().clone().unwrap();
        let grad_b2 = b2.grad().clone().unwrap();
        let grad_c2 = c2.grad().clone().unwrap();
        let grad_d2 = d2.grad().clone().unwrap();
        assert!(
            f64::abs(
                result1.grad().clone().unwrap().get_data()[0]
                    - result2.grad().clone().unwrap().get_data()[0]
            ) < 1e-10
        );
        assert!(
            f64::abs(
                result1.val().get_data()[0]
                    - (val_a * 2.0 + val_b * 2.0 + val_c + val_d).get_data()[0]
            ) < 1e-10
        );
        assert_eq!(grad_a1, grad_a2);
        assert_eq!(grad_a1.get_data()[0], 2.0);
        assert_eq!(grad_b1, grad_b2);
        assert_eq!(grad_b1.get_data()[0], 2.0);
        assert_eq!(grad_c1, grad_c2);
        assert_eq!(grad_c1.get_data()[0], 1.0);
        assert_eq!(grad_d1, grad_d2);
        assert_eq!(grad_d1.get_data()[0], 1.0);
    }
}

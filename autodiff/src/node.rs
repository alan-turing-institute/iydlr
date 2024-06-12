use std::{
    cell::RefCell,
    fmt::Display,
    ops::{Add, AddAssign, Deref, DerefMut, Div, Mul},
    rc::Rc,
};

use interfaces::{
    tensors::{Element, RealElement},
    utils::{Exp, Ln, Pow},
};
use num_traits::Zero;

// type Ptr<N> = Box<N>;
type Ptr<N> = Rc<RefCell<N>>;

// TODO: Rename to NodeContent
/// A node in a computation graph.
#[derive(Debug)]
pub enum NodeContent<T> {
    Sum(T, Option<T>, (Node<T>, Node<T>)),
    Prod(T, Option<T>, (Node<T>, Node<T>)),
    Exp(T, Option<T>, Node<T>),
    Ln(T, Option<T>, Node<T>),
    Pow(T, Option<T>, (Node<T>, Node<T>)),
    Leaf(T, Option<T>),
}

#[derive(Debug)]
pub struct Node<T> {
    ptr: Ptr<NodeContent<T>>,
}

// impl<T: RealElement + From<f64>> Deref for Node<T> {
//     type Target = NodeContent<T>;

//     fn deref(&self) -> &NodeContent<T> {
//         self.ptr.deref().borrow().deref()
//     }
// }

impl<T: RealElement + From<f64>> Node<T> {
    pub fn new(val: T, grad: Option<T>) -> Self {
        Node {
            ptr: Rc::new(RefCell::new(NodeContent::new(val, grad))),
        }
    }

    pub fn node(&self) -> NodeContent<T> {
        self.ptr.deref().borrow().deref().clone()
    }

    pub fn val(&self) -> T {
        self.ptr.deref().borrow().val().clone()
    }

    pub fn grad(&self) -> Option<T> {
        self.ptr.deref().borrow().grad().clone()
    }

    pub fn set_grad(&mut self, new_grad: T) {
        self.ptr.deref().borrow_mut().deref_mut().set_grad(new_grad)
    }

    pub fn add_assign_grad(&mut self, new_grad: T) {
        match self.grad() {
            Some(grad) => self.set_grad(grad + new_grad),
            None => self.set_grad(new_grad),
        }
    }

    // Set the gradient and initiate backward propagation.
    pub fn backward(mut self, gradient: T) -> Self {
        self.add_assign_grad(gradient);
        self.propagate_backward();
        self
    }

    // Propagate a given gradient on the `grad` of each associated Node.
    // Assumes the `grad` on self is not None.
    pub fn propagate_backward(&mut self) {
        // Unwrap safe because self.grad should have been assigned Some before descending:
        let self_grad = self.grad().unwrap();
        let self_val = self.val();

        match self.node() {
            NodeContent::Sum(_, _, (ref mut np1, ref mut np2)) => {
                np1.add_assign_grad(self_grad.clone());
                np2.add_assign_grad(self_grad);
                np1.propagate_backward();
                np2.propagate_backward();
            }
            NodeContent::Prod(_, _, (ref mut np1, ref mut np2)) => {
                np1.add_assign_grad(np2.val().to_owned() * self_grad.clone());
                np2.add_assign_grad(np1.val().to_owned() * self_grad);
                np1.propagate_backward();
                np2.propagate_backward();
            }
            NodeContent::Exp(_, _, ref mut np) => {
                np.add_assign_grad(self_grad * self_val);
                np.propagate_backward();
            }
            NodeContent::Ln(_, _, ref mut np) => {
                np.add_assign_grad(self_grad * <f64 as Into<T>>::into(1_f64) / np.val());
                np.propagate_backward();
            }
            // Node::Ln(_, _, ref mut n) => n.add_assign_grad(self_val.pow(<f64 as Into<T>>::into(-1_f64))),
            NodeContent::Pow(_, _, (ref mut np_b, ref mut np_e)) => {
                // exponent . base^(exponent - 1)
                let b_val = np_b.val().clone();
                let e_val = np_e.val().clone();
                let minus_one = <f64 as Into<T>>::into(-1_f64);
                np_b.add_assign_grad(
                    self_grad.clone()
                        * e_val.clone()
                        * b_val.clone().pow(e_val.clone() + minus_one),
                );

                // base^exponent . ln(base)
                np_e.add_assign_grad(self_grad * b_val.clone().pow(e_val.to_owned()) * b_val.ln());
                np_b.propagate_backward();
                np_e.propagate_backward();
            }
            NodeContent::Leaf(_, _) => {} // Do nothing.
        }
    }
}

impl<T> From<NodeContent<T>> for Node<T> {
    fn from(value: NodeContent<T>) -> Self {
        Node {
            ptr: Rc::new(RefCell::new(value)),
        }
    }
}

impl<T: RealElement + From<f64>> NodeContent<T> {
    pub fn new(val: T, grad: Option<T>) -> Self {
        NodeContent::Leaf(val, grad)
    }

    pub fn val(&self) -> &T {
        match self {
            NodeContent::Sum(val, _, _)
            | NodeContent::Prod(val, _, _)
            | NodeContent::Exp(val, _, _)
            | NodeContent::Ln(val, _, _)
            | NodeContent::Pow(val, _, _)
            | NodeContent::Leaf(val, _) => val,
        }
    }

    pub fn grad(&self) -> &Option<T> {
        match self {
            NodeContent::Sum(_, grad, _)
            | NodeContent::Prod(_, grad, _)
            | NodeContent::Exp(_, grad, _)
            | NodeContent::Ln(_, grad, _)
            | NodeContent::Pow(_, grad, _)
            | NodeContent::Leaf(_, grad) => grad,
        }
    }

    pub fn set_grad(&mut self, new_grad: T) {
        let g: &mut Option<T> = match self {
            NodeContent::Sum(_, grad, _)
            | NodeContent::Prod(_, grad, _)
            | NodeContent::Exp(_, grad, _)
            | NodeContent::Ln(_, grad, _)
            | NodeContent::Pow(_, grad, _)
            | NodeContent::Leaf(_, grad) => grad,
        };

        *g = Some(new_grad);
    }

    pub fn add_assign_grad(&mut self, new_grad: T) {
        let g: &mut Option<T> = match self {
            NodeContent::Sum(_, grad, _)
            | NodeContent::Prod(_, grad, _)
            | NodeContent::Exp(_, grad, _)
            | NodeContent::Ln(_, grad, _)
            | NodeContent::Pow(_, grad, _)
            | NodeContent::Leaf(_, grad) => grad,
        };

        match g {
            Some(grad) => *g = Some((*grad).clone() + new_grad),
            None => *g = Some(new_grad),
        }
    }
}

impl<T: RealElement + From<f64>> Add<NodeContent<T>> for NodeContent<T> {
    type Output = NodeContent<T>;

    fn add(self, rhs: NodeContent<T>) -> NodeContent<T> {
        NodeContent::Sum(
            self.val().clone() + rhs.val().clone(),
            None,
            (self.into(), rhs.into()),
        )
    }
}

impl<T: RealElement + From<f64>> Add<Node<T>> for Node<T> {
    type Output = Node<T>;

    fn add(self, rhs: Node<T>) -> Self::Output {
        NodeContent::Sum(self.val() + rhs.val(), None, (self, rhs)).into()
    }
}

impl<T: RealElement + From<f64>> Mul<NodeContent<T>> for NodeContent<T> {
    type Output = NodeContent<T>;

    fn mul(self, rhs: NodeContent<T>) -> NodeContent<T> {
        NodeContent::Prod(
            self.val().clone() * rhs.val().clone(),
            None,
            (self.into(), rhs.into()),
        )
    }
}

impl<T: RealElement + From<f64>> Mul<Node<T>> for Node<T> {
    type Output = Node<T>;

    fn mul(self, rhs: Node<T>) -> Self::Output {
        NodeContent::Prod(self.val() * rhs.val(), None, (self, rhs)).into()
    }
}

impl<T: RealElement + From<f64>> Div<NodeContent<T>> for NodeContent<T> {
    type Output = NodeContent<T>;

    fn div(self, rhs: NodeContent<T>) -> NodeContent<T> {
        // Same division by zero rules as standard division operator.
        NodeContent::Prod(
            self.val().clone() / rhs.val().clone(),
            None,
            (self.into(), rhs.into()),
        )
    }
}

impl<T: RealElement + From<f64>> Div<Node<T>> for Node<T> {
    type Output = Node<T>;

    fn div(self, rhs: Node<T>) -> Self::Output {
        NodeContent::Prod(self.val() / rhs.val(), None, (self, rhs)).into()
    }
}

impl<T: RealElement + From<f64>> Exp for NodeContent<T> {
    fn exp(self) -> Self {
        NodeContent::Exp(self.val().clone().exp(), None, self.into())
    }
}

impl<T: RealElement + From<f64>> Exp for Node<T> {
    fn exp(self) -> Self {
        NodeContent::Exp(self.val().exp(), None, self).into()
    }
}

impl<T: RealElement + From<f64>> Ln for NodeContent<T> {
    fn ln(self) -> Self {
        NodeContent::Exp(self.val().clone().ln(), None, self.into())
    }
}

impl<T: RealElement + From<f64>> Ln for Node<T> {
    fn ln(self) -> Self {
        NodeContent::Ln(self.val().ln(), None, self).into()
    }
}

impl<T: RealElement + From<f64>> Pow for NodeContent<T> {
    fn pow(self, exponent: NodeContent<T>) -> NodeContent<T> {
        NodeContent::Pow(
            self.val().clone().pow(exponent.val().clone()), // Note: unnecessary clone of exp.val() here?
            None,
            (self.into(), exponent.into()), // Base in position 1, exponent in position 2.
        )
    }
}

impl<T: RealElement + From<f64>> Pow for Node<T> {
    fn pow(self, exponent: Node<T>) -> Node<T> {
        NodeContent::Pow(self.val().pow(exponent.val()), None, (self, exponent)).into()
    }
}

impl<T: RealElement> AddAssign for NodeContent<T> {
    fn add_assign(&mut self, _rhs: Self) {
        panic!("Unexpected call to AddAssign on a Node.")
    }
}

impl<T: RealElement> AddAssign for Node<T> {
    fn add_assign(&mut self, _rhs: Self) {
        panic!("Unexpected call to AddAssign on a Node.")
    }
}

impl<T: RealElement> Display for NodeContent<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node: {:?}", self)
    }
}

impl<T: RealElement> Display for Node<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node: {:?}", self.ptr.deref().borrow())
    }
}

// TODO: check this (was auto-generated). Is it what we want?
impl<T: RealElement> Clone for NodeContent<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Sum(arg0, arg1, arg2) => Self::Sum(arg0.clone(), arg1.clone(), arg2.clone()),
            Self::Prod(arg0, arg1, arg2) => Self::Prod(arg0.clone(), arg1.clone(), arg2.clone()),
            Self::Exp(arg0, arg1, arg2) => Self::Exp(arg0.clone(), arg1.clone(), arg2.clone()),
            Self::Ln(arg0, arg1, arg2) => Self::Ln(arg0.clone(), arg1.clone(), arg2.clone()),
            Self::Pow(arg0, arg1, arg2) => Self::Pow(arg0.clone(), arg1.clone(), arg2.clone()),
            Self::Leaf(arg0, arg1) => Self::Leaf(arg0.clone(), arg1.clone()),
        }
    }
}

impl<T: RealElement> Clone for Node<T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr.clone(),
        }
    }
}

impl<T: RealElement> PartialEq for NodeContent<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Sum(l0, l1, l2), Self::Sum(r0, r1, r2)) => l0 == r0 && l1 == r1 && l2 == r2,
            (Self::Prod(l0, l1, l2), Self::Prod(r0, r1, r2)) => l0 == r0 && l1 == r1 && l2 == r2,
            (Self::Exp(l0, l1, l2), Self::Exp(r0, r1, r2)) => l0 == r0 && l1 == r1 && l2 == r2,
            (Self::Ln(l0, l1, l2), Self::Ln(r0, r1, r2)) => l0 == r0 && l1 == r1 && l2 == r2,
            (Self::Pow(l0, l1, l2), Self::Pow(r0, r1, r2)) => l0 == r0 && l1 == r1 && l2 == r2,
            (Self::Leaf(l0, l1), Self::Leaf(r0, r1)) => l0 == r0 && l1 == r1,
            _ => false,
        }
    }
}

impl<T: RealElement> PartialEq for Node<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

// impl<T: RealElement> From<T> for Node<T> {
//     fn from(value: T) -> Self {
//         Node::new(value, None)
//     }
// }

impl<T: RealElement> From<f64> for NodeContent<T> {
    fn from(value: f64) -> Self {
        NodeContent::new(value.into(), None)
    }
}

impl<T: RealElement> From<f64> for Node<T> {
    fn from(value: f64) -> Self {
        NodeContent::<T>::from(value).into()
    }
}

impl<T: RealElement> Zero for NodeContent<T> {
    fn zero() -> Self {
        NodeContent::new(0_f64.into(), None)
    }

    fn is_zero(&self) -> bool {
        Self::zero().eq(self)
    }
}

impl<T: RealElement> Zero for Node<T> {
    fn zero() -> Self {
        NodeContent::<T>::zero().into()
    }

    fn is_zero(&self) -> bool {
        Self::zero().eq(self)
    }
}

impl<T: RealElement + From<f64>> Element for NodeContent<T> {}

impl<T: RealElement + From<f64>> RealElement for NodeContent<T> {
    fn neg_inf() -> Self {
        NodeContent::new((-f64::INFINITY).into(), None)
    }
}

impl<T: RealElement + From<f64>> Element for Node<T> {}

impl<T: RealElement + From<f64>> RealElement for Node<T> {
    fn neg_inf() -> Self {
        NodeContent::<T>::neg_inf().into()
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_new() {
        let node = NodeContent::<f64>::new(3.1, Some(0.4));
        assert_eq!(node.val(), &3.1_f64);
        assert_eq!(node.grad(), &Some(0.4));
    }

    #[test]
    fn test_set_grad() {
        let mut node = NodeContent::<f64>::new(3.1, None);
        assert_eq!(node.val(), &3.1_f64);
        assert_eq!(node.grad(), &None);

        node.set_grad(0.4);

        assert_eq!(node.val(), &3.1_f64);
        assert_eq!(node.grad(), &Some(0.4));

        node.set_grad(0.7);

        assert_eq!(node.val(), &3.1_f64);
        assert_eq!(node.grad(), &Some(0.7));
    }

    #[test]
    fn test_set_grad_node_ptr() {
        let mut node = Node::new(3.1, None);
        assert_eq!(node.val(), 3.1_f64);
        assert_eq!(node.grad(), None);

        let node_clone = node.clone();
        assert_eq!(node_clone.val(), 3.1_f64);
        assert_eq!(node_clone.grad(), None);

        node.set_grad(0.4);

        assert_eq!(node.val(), 3.1_f64);
        assert_eq!(node.grad(), Some(0.4));

        assert_eq!(node_clone.val(), 3.1_f64);
        assert_eq!(node_clone.grad(), Some(0.4));

        node.set_grad(0.7);

        assert_eq!(node.val(), 3.1_f64);
        assert_eq!(node.grad(), Some(0.7));

        assert_eq!(node_clone.val(), 3.1_f64);
        assert_eq!(node_clone.grad(), Some(0.7));
    }

    #[test]
    fn test_add_assign_grad() {
        let mut node = NodeContent::<f64>::new(3.1, None);
        assert_eq!(node.val(), &3.1_f64);
        assert_eq!(node.grad(), &None);

        node.add_assign_grad(0.4);

        assert_eq!(node.val(), &3.1_f64);
        assert_eq!(node.grad(), &Some(0.4));

        node.add_assign_grad(0.7);

        assert_eq!(node.val(), &3.1_f64);
        assert_eq!(node.grad(), &Some(1.1));
    }

    #[test]
    fn test_add_assign_grad_node_ptr() {
        let mut node = Node::new(3.1, None);

        assert_eq!(node.val(), 3.1_f64);
        assert_eq!(node.grad(), None);

        node.add_assign_grad(0.4);

        assert_eq!(node.val(), 3.1_f64);
        assert_eq!(node.grad(), Some(0.4));

        node.add_assign_grad(0.7);

        assert_eq!(node.val(), 3.1_f64);
        assert_eq!(node.grad(), Some(1.1));
    }

    #[test]
    fn test_add() {
        let node1 = NodeContent::<f64>::new(3.1, Some(0.4));
        let node2 = NodeContent::<f64>::new(22.2, None);

        let result = node1 + node2;
        assert_eq!(result.val(), &25.3_f64);
        assert_eq!(result.grad(), &None);
    }

    #[test]
    fn test_add_node_ptr() {
        let node1 = NodeContent::<f64>::new(3.1, Some(0.4));
        let node2 = NodeContent::<f64>::new(22.2, None);

        let np1: Node<f64> = node1.into();
        let np2 = node2.into();

        let result = np1 + np2;
        assert_eq!(result.val(), 25.3_f64);
        assert_eq!(result.grad(), None);
    }

    #[test]
    fn test_mul() {
        let node1 = NodeContent::<f64>::new(3.1, Some(0.4));
        let node2 = NodeContent::<f64>::new(22.2, None);

        let result = node1 * node2;
        assert_eq!(result.val(), &68.82_f64);
        assert_eq!(result.grad(), &None);
    }

    #[test]
    fn test_mul_node_ptr() {
        let node1 = NodeContent::<f64>::new(3.1, Some(0.4));
        let node2 = NodeContent::<f64>::new(22.2, None);

        let np1: Node<f64> = node1.into();
        let np2 = node2.into();

        let result = np1 * np2;
        assert_eq!(result.val(), 68.82_f64);
        assert_eq!(result.grad(), None);
    }

    #[test]
    fn test_div() {
        let node1 = NodeContent::<f64>::new(3.1, Some(0.4));
        let node2 = NodeContent::<f64>::new(22.2, None);

        let result = node1 / node2;
        assert_eq!(result.val(), &0.13963963963963966_f64);
        assert_eq!(result.grad(), &None);
    }

    #[test]
    fn test_div_node_ptr() {
        let node1 = NodeContent::<f64>::new(3.1, Some(0.4));
        let node2 = NodeContent::<f64>::new(22.2, None);

        let np1: Node<f64> = node1.into();
        let np2 = node2.into();

        let result = np1 / np2;
        assert_eq!(result.val(), 0.13963963963963966_f64);
        assert_eq!(result.grad(), None);
    }

    #[test]
    fn test_div_by_zero() {
        let node1 = NodeContent::<f64>::new(3.1, Some(0.4));
        let node2 = NodeContent::<f64>::new(0.0, None);

        let result = node1 / node2;
        assert_eq!(result.val(), &f64::INFINITY);
    }

    #[test]
    fn test_pow() {
        let node1 = NodeContent::<f64>::new(3.1, Some(0.4));
        let node2 = NodeContent::<f64>::new(22.2, None);

        let result = node1.pow(node2);
        assert_eq!(result.val(), &80952376567.60643_f64);
        assert_eq!(result.grad(), &None);
    }

    #[test]
    fn test_pow_node_ptr() {
        let node1 = NodeContent::<f64>::new(3.1, Some(0.4));
        let node2 = NodeContent::<f64>::new(22.2, None);

        let np1: Node<f64> = node1.into();
        let np2 = node2.into();

        let result = np1.pow(np2);
        assert_eq!(result.val(), 80952376567.60643_f64);
        assert_eq!(result.grad(), None);
    }
    // #[test]
    // fn test_fmt() {
    //     let node = Node::<f64>::new(3.1, None);
    //     println!("{}", node);
    // }

    #[test]
    fn test_backward_on_sum() {
        let node1 = Node::new(1.1, None);
        let node2 = Node::new(2.2, None);

        let node = node1.clone() + node2.clone();

        assert!(node.grad().is_none());
        assert!(node1.grad().is_none());
        assert!(node2.grad().is_none());

        let node = node.backward(5.0);

        assert!(node.grad().is_some());
        assert_eq!(node.grad().unwrap(), 5.0_f64);

        assert!(node1.grad().is_some());
        assert_eq!(node1.grad().unwrap(), 5.0_f64);
        assert!(node2.grad().is_some());
        assert_eq!(node2.grad().unwrap(), 5.0_f64);
    }

    #[test]
    fn test_backward_on_prod() {
        let node1 = Node::new(1.1, None);
        let node2 = Node::new(2.2, None);

        let node = node1.clone() * node2.clone();

        assert!(node.grad().is_none());
        assert!(node1.grad().is_none());
        assert!(node2.grad().is_none());

        let node = node.backward(5.0);

        assert!(node.grad().is_some());
        assert_eq!(node.grad().unwrap(), 5.0_f64);

        assert!(node1.grad().is_some());
        assert_eq!(node1.grad().unwrap(), 11.0_f64);
        assert!(node2.grad().is_some());
        assert_eq!(node2.grad().unwrap(), 5.5_f64);
    }

    #[test]
    fn test_backward_on_prod_sum() {
        let node_a = Node::new(3.0, None);
        let node_b = Node::new(2.0, None);
        let node_c = Node::new(2.0, None);

        let node_d = node_a.clone() + node_b.clone();
        let node_f = node_d.clone() * node_c.clone();

        // Check all grads are None initially.
        assert!(node_f.grad().is_none());
        assert!(node_a.grad().is_none());
        assert!(node_b.grad().is_none());
        assert!(node_c.grad().is_none());
        assert!(node_d.grad().is_none());

        let node_f = node_f.backward(10.0);

        // Check all grads have been populated.
        assert!(node_f.grad().is_some());
        assert_eq!(node_f.grad().unwrap(), 10.0_f64);

        assert!(node_d.grad().is_some());
        assert_eq!(node_d.grad().unwrap(), 20.0_f64);
        assert!(node_c.grad().is_some());
        assert_eq!(node_c.grad().unwrap(), 50.0_f64);

        assert!(node_a.grad().is_some());
        assert_eq!(node_a.grad().unwrap(), 20.0_f64);
        assert!(node_b.grad().is_some());
        assert_eq!(node_b.grad().unwrap(), 20.0_f64);
    }

    #[test]
    fn test_backward_on_2x_squared_plus_exp_5x() {
        // Expression: f(x) = 2x^2 + exp(5x)

        let node_x = Node::new(3.0, None);

        let node_2 = Node::new(2.0, None);
        let node_2_ = Node::new(2.0, None);
        let node_5 = Node::new(5.0, None);

        let node_5x = node_5.clone() * node_x.clone();
        let node_exp_5x = node_5x.clone().exp();
        let node_x_squared = node_x.clone().pow(node_2.clone());
        let node_2x_squared = node_x_squared.clone() * node_2_.clone();

        let node_f = node_exp_5x.clone() + node_2x_squared.clone();

        let node_f = node_f.backward(1.0);

        // Check all grads have been populated.
        assert_eq!(node_f.grad().unwrap(), 1.0_f64);

        // w.r.t. the exp(5x) node the grad is 1.
        assert_eq!(node_exp_5x.grad().unwrap(), 1.0_f64);

        // w.r.t. the 2x^2 node the grad is 1.
        assert_eq!(node_2x_squared.grad().unwrap(), 1.0_f64);

        // w.r.t. the x^2 node the grad is 2.
        assert_eq!(node_x_squared.grad().unwrap(), 2.0_f64);

        // w.r.t. the 5x node the grad is exp(5*3)
        assert_eq!(node_5x.grad().unwrap(), 3269017.372472110639302_f64);

        // w.r.t. the 2 node that is the exponent of x^2, the grad is 2 * 3^2 * ln(3) = 19.775021196025975.
        assert_eq!(node_2.grad().unwrap(), 19.775021196025975_f64);

        // w.r.t. the 2 node that multiplies the x^2, the grad is 3^2:
        assert_eq!(node_2_.grad().unwrap(), 9.0_f64);

        // w.r.t. the 5 node that multiplies the x, the grad is 3 * e^(5*3) =
        assert_eq!(node_5.grad().unwrap(), 9807052.117416331917906_f64);

        // df/dx = 4x + 5 exp(5x)
        // So, w.r.t. the x node, the grad is df/dx(3) = 12 + 5 * exp(15) = 16345098.862360553196509
        assert_eq!(node_x.grad().unwrap(), 16345098.862360553196509_f64);
    }

    #[test]
    fn test_overly_complicated_identity_function() {
        // Expression: f(x) = exp(
        //  x*(
        //    ln(
        //     (x+3.14)
        //    )/x
        //   )
        // ) - 3.14 = x
        let node_x = Node::new(3.0, None);
        let node_314 = Node::new(3.14, None);
        let node_1 = node_x.clone() + node_314.clone();
        let node_1 = node_1.clone().ln();
        let node_1 = node_1.clone() * node_x.clone().pow(Node::from(-1.0));
        let node_1 = node_1.clone() * node_x.clone();
        let node_1 = node_1.clone().exp();
        let node_1 = node_1.clone() + (Node::from(-1.0) * node_314.clone());
        let node_1 = node_1.backward(1.0);
        assert!(f64::abs(node_x.grad().unwrap() - 1.0_f64) < 1e-10);
    }
}

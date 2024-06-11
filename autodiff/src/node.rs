use std::{
    cell::RefCell,
    fmt::Display,
    ops::{Add, AddAssign, Deref, Div, Mul},
};

use interfaces::{
    tensors::{Element, RealElement},
    utils::{Exp, Ln, Pow},
};

type Ptr<N> = Box<N>;
// type Ptr<N> = Rc<RefCell<N>>;

/// A node in a computation graph.
#[derive(Debug)]
pub enum Node<T> {
    // Replace Box<Node<T>> with Rc<Node<T>> if/when we need multiple ownership of nodes/subgraphs.
    Sum(T, Option<T>, (Ptr<Node<T>>, Ptr<Node<T>>)),
    Prod(T, Option<T>, (Ptr<Node<T>>, Ptr<Node<T>>)),
    Exp(T, Option<T>, Ptr<Node<T>>),
    Ln(T, Option<T>, Ptr<Node<T>>),
    Pow(T, Option<T>, (Ptr<Node<T>>, Ptr<Node<T>>)),
    Leaf(T, Option<T>),
}

impl<T: RealElement + From<f64>> Node<T> {
    pub fn new(val: T, grad: Option<T>) -> Self {
        Node::Leaf(val, grad)
    }

    pub fn val(&self) -> &T {
        match self {
            Node::Sum(val, _, _)
            | Node::Prod(val, _, _)
            | Node::Exp(val, _, _)
            | Node::Ln(val, _, _)
            | Node::Pow(val, _, _)
            | Node::Leaf(val, _) => val,
        }
    }

    pub fn grad(&self) -> &Option<T> {
        match self {
            Node::Sum(_, grad, _)
            | Node::Prod(_, grad, _)
            | Node::Exp(_, grad, _)
            | Node::Ln(_, grad, _)
            | Node::Pow(_, grad, _)
            | Node::Leaf(_, grad) => grad,
        }
    }

    // TODO: update to more like add_assign than overwrite.
    pub fn set_grad(&mut self, new_grad: T) {
        let g = match self {
            Node::Sum(_, grad, _)
            | Node::Prod(_, grad, _)
            | Node::Exp(_, grad, _)
            | Node::Ln(_, grad, _)
            | Node::Pow(_, grad, _)
            | Node::Leaf(_, grad) => grad,
        };

        *g = Some(new_grad);
    }

    // Set the gradient and initiate backward propagation.
    pub fn backward(mut self, gradient: T) -> Self {
        self.set_grad(gradient);
        self.propagate_backward();
        self
    }

    // Propagate a given gradient on the `grad` of each associated Node.
    // Assumes the `grad` on self is not None.
    pub fn propagate_backward(&mut self) {
        let self_val = self.val().clone();
        let self_grad = <Option<T> as Clone>::clone(&self.grad()).unwrap();

        // TODO: check all these: why is there a factor self_grad in Sum & Prod but not elsewhere?
        match self {
            Node::Sum(_, _, (ref mut n1, ref mut n2)) => {
                n1.set_grad(self_grad.to_owned());
                n2.set_grad(self_grad.to_owned());
                n1.propagate_backward();
                n2.propagate_backward(); // TODO: spawn new thread.
            }
            Node::Prod(_, _, (ref mut n1, ref mut n2)) => {
                n1.set_grad(n2.val().to_owned() * self_grad.clone());
                n2.set_grad(n1.val().to_owned() * self_grad);
                n1.propagate_backward();
                n2.propagate_backward(); // TODO: spawn new thread.
            }
            Node::Exp(_, _, ref mut n) => {
                n.set_grad(self_val);
                n.propagate_backward();
            }
            Node::Ln(_, _, ref mut n) => {
                n.set_grad(<f64 as Into<T>>::into(1_f64) / self_val);
                n.propagate_backward();
            }
            // Node::Ln(_, _, ref mut n) => n.set_grad(self_val.pow(<f64 as Into<T>>::into(-1_f64))),
            Node::Pow(_, _, (ref mut n1, ref mut n2)) => {
                // exponent . base^(exponent - 1)
                n1.set_grad(
                    n2.val().to_owned()
                        * n1.val()
                            .clone()
                            .pow(n2.val().to_owned() + <f64 as Into<T>>::into(-1_f64)),
                );
                // base^exponent . ln(base)
                n2.set_grad((n1.val().clone()).pow(n2.val().to_owned()) * n1.val().clone().ln());
                n1.propagate_backward();
                n2.propagate_backward(); // TODO: spawn new thread.
            }
            Node::Leaf(_, _) => {} // Do nothing.
        }
    }
}

impl<T: RealElement + From<f64>> Add<Node<T>> for Node<T> {
    type Output = Node<T>;

    fn add(self, _rhs: Node<T>) -> Node<T> {
        Node::Sum(
            self.val().clone() + _rhs.val().clone(),
            None,
            (self.into(), _rhs.into()),
        )
    }
}

impl<T: RealElement + From<f64>> Mul<Node<T>> for Node<T> {
    type Output = Node<T>;

    fn mul(self, _rhs: Node<T>) -> Node<T> {
        Node::Prod(
            self.val().clone() * _rhs.val().clone(),
            None,
            (self.into(), _rhs.into()),
        )
    }
}

impl<T: RealElement + From<f64>> Div<Node<T>> for Node<T> {
    type Output = Node<T>;

    fn div(self, _rhs: Node<T>) -> Node<T> {
        // Same division by zero rules as standard division operator.
        Node::Prod(
            self.val().clone() / _rhs.val().clone(),
            None,
            (self.into(), _rhs.into()),
        )
    }
}

impl<T: RealElement + From<f64>> Exp for Node<T> {
    fn exp(self) -> Self {
        Node::Exp(self.val().clone().exp(), None, self.into())
    }
}

impl<T: RealElement + From<f64>> Ln for Node<T> {
    fn ln(self) -> Self {
        Node::Exp(self.val().clone().ln(), None, self.into())
    }
}

impl<T: RealElement + From<f64>> Pow for Node<T> {
    fn pow(self, exponent: Node<T>) -> Node<T> {
        Node::Pow(
            self.val().clone().pow(exponent.val().clone()), // Note: unnecessary clone of exp.val() here?
            None,
            (self.into(), exponent.into()), // Base in position 1, exponent in position 2.
        )
    }
}

impl<T: RealElement> AddAssign for Node<T> {
    fn add_assign(&mut self, _rhs: Self) {
        panic!("Unexpected call to AddAssign on a Node.")
    }
}

impl<T: RealElement> Display for Node<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node: {:?}", self)
    }
}

impl<T: RealElement> Clone for Node<T> {
    fn clone(&self) -> Self {
        todo!();
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

impl<T: RealElement + From<f64>> Element for Node<T> {}
impl<T: RealElement + From<f64>> RealElement for Node<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let node = Node::<f64>::new(3.1, Some(0.4));
        assert_eq!(node.val(), &3.1_f64);
        assert_eq!(node.grad(), &Some(0.4));
    }

    #[test]
    fn test_set_grad() {
        let mut node = Node::<f64>::new(3.1, None);
        assert_eq!(node.val(), &3.1_f64);
        assert_eq!(node.grad(), &None);

        node.set_grad(0.4);

        assert_eq!(node.val(), &3.1_f64);
        assert_eq!(node.grad(), &Some(0.4));
    }

    #[test]
    fn test_add() {
        let node1 = Node::<f64>::new(3.1, Some(0.4));
        let node2 = Node::<f64>::new(22.2, None);

        let result = node1 + node2;
        assert_eq!(result.val(), &25.3_f64);
        assert_eq!(result.grad(), &None);
    }

    #[test]
    fn test_mul() {
        let node1 = Node::<f64>::new(3.1, Some(0.4));
        let node2 = Node::<f64>::new(22.2, None);

        let result = node1 * node2;
        assert_eq!(result.val(), &68.82_f64);
        assert_eq!(result.grad(), &None);
    }

    #[test]
    fn test_div() {
        let node1 = Node::<f64>::new(3.1, Some(0.4));
        let node2 = Node::<f64>::new(22.2, None);

        let result = node1 / node2;
        assert_eq!(result.val(), &0.13963963963963966_f64);
        assert_eq!(result.grad(), &None);
    }

    #[test]
    fn test_div_by_zero() {
        let node1 = Node::<f64>::new(3.1, Some(0.4));
        let node2 = Node::<f64>::new(0.0, None);

        let result = node1 / node2;
        assert_eq!(result.val(), &f64::INFINITY);
    }

    #[test]
    fn test_pow() {
        let node1 = Node::<f64>::new(3.1, Some(0.4));
        let node2 = Node::<f64>::new(22.2, None);

        let result = node1.pow(node2);
        assert_eq!(result.val(), &80952376567.60643_f64);
        assert_eq!(result.grad(), &None);
    }

    // #[test]
    // fn test_fmt() {
    //     let node = Node::<f64>::new(3.1, None);
    //     println!("{}", node);
    // }

    #[test]
    fn test_backward() {
        let node1 = Node::new(1.1, None);
        let node2 = Node::new(2.2, None);

        let node = node1 + node2;

        assert!(node.grad().is_none());
        match &node {
            Node::Sum(_, _, (n1, n2)) => {
                assert!(n1.grad().is_none());
                assert!(n2.grad().is_none());
            }
            _ => panic!(),
        }

        let node = node.backward(5.0);

        assert!(node.grad().is_some());
        assert_eq!(node.grad().unwrap(), 5.0_f64);
        match &node {
            Node::Sum(_, _, (n1, n2)) => {
                assert!(n1.grad().is_some());
                assert_eq!(n1.grad().unwrap(), 5.0_f64);
                assert!(n2.grad().is_some());
                assert_eq!(n2.grad().unwrap(), 5.0_f64);
            }
            _ => panic!(),
        }
    }
}

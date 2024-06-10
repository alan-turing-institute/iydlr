// /// A node in a computation graph.
// pub struct Node<T> {
//     val: T,          // value
//     grad: Option<T>, // gradient
//     dep: Vec<Node>,  // dependencies
// }

use std::{
    ops::{self, Add},
    rc::Rc,
};

use interfaces::tensors::RealElement;

/// A node in a computation graph.
pub enum Node<T> {
    Sum(T, Option<T>, (Rc<Node<T>>, Rc<Node<T>>)),
    Prod(T, Option<T>, (Rc<Node<T>>, Rc<Node<T>>)),
    Exp(T, Option<T>, Rc<Node<T>>),
    Ln(T, Option<T>, Rc<Node<T>>),
    Pow(T, Option<T>, (Rc<Node<T>>, Rc<Node<T>>)),
    Leaf(T, Option<T>),
}

impl<T: Add> Node<T> {
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

    pub fn grad(&self) -> Option<&T> {
        match self {
            Node::Sum(_, grad, _)
            | Node::Prod(_, grad, _)
            | Node::Exp(_, grad, _)
            | Node::Ln(_, grad, _)
            | Node::Pow(_, grad, _)
            | Node::Leaf(_, grad) => grad.as_ref(),
        }
    }
}

impl<T: RealElement> ops::Add<Node<T>> for Node<T> {
    type Output = Node<T>;

    fn add(self, _rhs: Node<T>) -> Node<T> {
        Node::Sum(
            self.val().clone() + _rhs.val().clone(),
            None,
            (self.into(), _rhs.into()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let node = Node::<f64>::new(3.1, Some(0.4));
        assert_eq!(node.val(), &3.1_f64);
        assert_eq!(node.grad(), Some(&0.4));
    }

    #[test]
    fn test_add() {
        let node1 = Node::<f64>::new(3.1, Some(0.4));
        let node2 = Node::<f64>::new(22.2, None);

        let result = node1 + node2;
        assert_eq!(result.val(), &25.3_f64);
        assert_eq!(result.grad(), None);
    }
}

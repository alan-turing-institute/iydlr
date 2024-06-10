/// A node in a computation graph.
pub struct Node<T> {
    val: T,
    grad: Option<T>,
}

impl<T> Node<T> {
    pub fn new(val: T) -> Self {
        Node {
            val: val,
            grad: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let node = Node::<f64>::new(3.1);
        assert_eq!(node.val, 3.1f64);
        assert_eq!(node.grad, None);
    }
}

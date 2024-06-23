use autodiff::node::Node;
use interfaces::tensors::Tensor;
use tensors::TensorImpl;

#[test]
fn test_autodiff_tensor() {
    let batch_size = 3;
    let input_dim = 2;
    let output_dim = 4;
    let w = Node::new(
        TensorImpl::from_vec(
            &vec![input_dim, output_dim],
            &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap(),
        None,
    );
    let x = Node::new(
        TensorImpl::from_vec(
            &vec![batch_size, input_dim],
            &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap(),
        None,
    );
    let b = Node::new(
        TensorImpl::from_vec(&vec![1, output_dim], &vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        None,
    );

    let mut y = x.matmul(&w).unwrap() + b;
    println!("y shape: {:?}", y.shape());
    y.backward(TensorImpl::fill_with_clone(vec![3, 4], 1.0));
    println!("{:?}", x.grad().clone().unwrap());
}

use autodiff::node::Node;
use interfaces::deep_learning::DLModule;
use interfaces::tensors::RealTensor;
use interfaces::tensors::Tensor;
use neural_nets::optim::bce;
use neural_nets::{
    act_layer::ActLayer, lin_layer::LinLayer, optim::OptimSGD, serial::Serial,
    xor_generator::XorGenerator,
};
use tensors::TensorImpl;

#[test]
fn xor_test() {
    let seed = 2;
    let max_itr = 300;
    let batch_size = 5;
    let model: Serial<TensorImpl<Node<f64>>, Node<f64>> = Serial::new(vec![
        Box::new(LinLayer::new(2, 5, seed)),
        Box::new(ActLayer::new()),
        Box::new(LinLayer::new(5, 10, seed)),
        Box::new(ActLayer::new()),
        Box::new(LinLayer::new(10, 10, seed)),
        Box::new(ActLayer::new()),
        Box::new(LinLayer::new(10, 2, seed)),
    ]);

    let mut xor_gen = XorGenerator::new(batch_size, seed);

    let mut optim = OptimSGD::new(0.01, max_itr, model.params());

    for itr in 0..max_itr {
        let (x, y) = xor_gen.next().unwrap();
        let y_tensor = TensorImpl::from_vec(&vec![1, batch_size, 1], &y).unwrap();
        let pred = model.forward(&x).unwrap();
        let soft = pred.softmax(2);
        let class_0 = soft
            .matmul(
                &TensorImpl::from_vec(&vec![2, 1], &vec![Node::from(1.0), Node::from(0.0)])
                    .unwrap(),
            )
            .unwrap();

        let loss_tensor = bce(y_tensor, class_0);
        let loss = loss_tensor.dim_sum(vec![1]);

        println!(
            "loss: {:?}",
            loss.clone()
                .into_iter()
                .map(|node| node.val())
                .collect::<Vec<_>>()
        );

        optim.zero_grad();
        loss.at(vec![0, 0, 0]).unwrap().clone().backward(1.0);
        optim.update(itr);
    }
    let (x, y) = xor_gen.next().unwrap();
    let y_tensor = TensorImpl::from_vec(&vec![1, batch_size, 1], &y).unwrap();
    // shape (1,B,1)
    let pred = model.forward(&x).unwrap();
    let soft = pred.softmax(2);
    let class_0 = soft
        .matmul(
            &TensorImpl::from_vec(&vec![2, 1], &vec![Node::from(1.0), Node::from(0.0)]).unwrap(),
        )
        .unwrap();
    println!(
        "class_0: {:?}",
        class_0
            .clone()
            .into_iter()
            .map(|node| node.val())
            .collect::<Vec<_>>()
    );
    println!(
        "truth: {:?}",
        y_tensor
            .clone()
            .into_iter()
            .map(|node| node.val())
            .collect::<Vec<_>>()
    );

    let loss_tensor = bce(y_tensor, class_0);
    let loss = loss_tensor.dim_sum(vec![1]);
    assert!(loss.at(vec![0, 0, 0]).unwrap().clone().val() < 0.1_f64)
}

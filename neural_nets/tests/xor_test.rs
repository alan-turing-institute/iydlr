use interfaces::{deep_learning::DLModule, tensors::Tensor};
use neural_nets::{lin_layer::LinLayer, optim::OptimSGD, serial::Serial};
use tensors::TensorImpl;

#[test]
fn xor_test() {
    let seed = 0;
    let max_itr = 2;
    let model: Serial<TensorImpl<f64>, f64> = Serial::new(vec![
        Box::new(LinLayer::new(2, 5, seed)),
        Box::new(LinLayer::new(5, 10, seed)),
    ]);

    let x = TensorImpl::from_vec(&vec![1, 4, 2], &vec![7.0; 8]).unwrap();

    let optim = OptimSGD::new(0.01, max_itr, model.params());

    for _ in 0..max_itr {
        let pred = model.forward(&x).unwrap();
        println!("{:?}", pred);
    }
}

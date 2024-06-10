use std::marker::PhantomData;

use interfaces::deep_learning::LinearLayer;
use interfaces::tensors::{Element, Tensor};

///
///

// struct Attention<T, E, Q, K, V, M>
pub struct Attention<T, E, Q, K, V, M>
where
    Q: LinearLayer<T, E>,
    K: LinearLayer<T, E>,
    V: LinearLayer<T, E>,
    M: Tensor<E>,
    T: Tensor<E>,
    E: Element,
{
    pub query: Q,
    pub key: K,
    pub value: V,
    pub mask: M,
    pub _marker_t: PhantomData<T>,
    pub _marker_e: PhantomData<E>,
}

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

impl OptimSGD<f64> {
    pub fn zero_grad(&self) {
        todo!();
        // for p in self.params.iter() {
        //     p.zero_grad()
        // }
    }

    pub fn update(&self, itr: usize) {
        let mut l_rate = self.l_rate;
        if itr > self.max_itr.saturating_mul(3).saturating_div(4) {
            l_rate *= 0.1;
        }
        todo!();
        // for p in self.params.iter() {
        //     p.add_data(-l_rate * p.grad())
        // }
    }
}

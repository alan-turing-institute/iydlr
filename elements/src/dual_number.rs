use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign},
};

use interfaces::{
    tensors::{Element, RealElement},
    utils::{Exp, Ln, Pow},
};
use num_traits::identities::Zero;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DualNumber {
    pub real: f64,
    pub dual: f64,
}

impl Display for DualNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.real, self.dual)
    }
}

impl DualNumber {
    pub fn new(real: f64, dual: f64) -> Self {
        Self { real, dual }
    }
}

impl Add for DualNumber {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.real + rhs.real, self.dual + rhs.dual)
    }
}

impl AddAssign for DualNumber {
    fn add_assign(&mut self, rhs: Self) {
        self.real += rhs.real;
        self.dual += rhs.dual;
    }
}

impl Mul for DualNumber {
    type Output = Self;
    //  (ax + (ay + xb) i) =  (x + iy) * (a + bi)
    // Problem in Mul?
    fn mul(self, rhs: Self) -> Self::Output {
        let real = self.real * rhs.real;
        let dual = self.real * rhs.dual + self.dual * rhs.real;
        Self::new(real, dual)
    }
}

impl MulAssign for DualNumber {
    fn mul_assign(&mut self, rhs: Self) {
        self.real *= rhs.real;
        self.dual = self.real * rhs.dual + self.dual * rhs.real;
    }
}

impl Div for DualNumber {
    type Output = Self;
    //  (x + ye) / (a + be)
    //  = (x + ye)(a - be) / ((a + be)(a - be))
    //  = (ax + (ay - xb)e) / (a**2)
    fn div(self, rhs: Self) -> Self::Output {
        let denom = self.real * self.real;
        let real = (self.real * rhs.real) / denom;
        let dual = (self.real * rhs.dual - self.dual * rhs.real) / denom;
        Self::new(real, dual)
    }
}

impl DivAssign for DualNumber {
    fn div_assign(&mut self, rhs: Self) {
        let denom = self.real * self.real;
        self.real = (self.real * rhs.real) / denom;
        self.dual = (self.real * rhs.dual - self.dual * rhs.real) / denom;
    }
}

// f(a + be) = f(a) + f'(a)be
impl Exp for DualNumber {
    fn exp(self) -> Self {
        let real = self.real.exp();
        let dual = self.dual * self.real.exp();
        Self::new(real, dual)
    }
}

impl Ln for DualNumber {
    fn ln(self) -> Self {
        let real = self.real.ln();
        let dual = self.dual / self.real;
        Self::new(real, dual)
    }
}

impl Pow for DualNumber {
    fn pow(self, exp: Self) -> Self {
        let real = self.real.powf(exp.real);
        let dual = real * (exp.dual * self.real.ln() + (self.dual * exp.real / self.real));
        Self::new(real, dual)
    }
}

impl Zero for DualNumber {
    fn zero() -> Self {
        Self::new(0., 0.)
    }
    fn is_zero(&self) -> bool {
        self.real == 0. && self.dual == 0.
    }
}

impl Element for DualNumber {}

fn grad<F: Fn(DualNumber) -> DualNumber>(func: F, value: f64) -> f64 {
    func(DualNumber::new(value, 1.)).dual
}

impl DualNumber {
    fn grad<F: Fn(DualNumber) -> DualNumber>(func: F, value: f64) -> f64 {
        func(DualNumber::new(value, 1.)).dual
    }
}

pub trait Grad {
    fn grad(&self, value: f64) -> f64;
}
impl<F> Grad for F
where
    F: Fn(DualNumber) -> DualNumber,
{
    fn grad(&self, value: f64) -> f64 {
        grad(self, value)
    }
}

impl RealElement for DualNumber {
    fn neg_inf() -> Self {
        Self::new(-f64::INFINITY, 0.)
    }
}

impl From<f64> for DualNumber {
    fn from(value: f64) -> Self {
        Self::new(value, 0.)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_dual() {
        let dual_a = DualNumber::new(3., 4.);
        let dual_b = DualNumber::new(2., 3.);
        let dual_mul = dual_a * dual_b;
        assert_eq!(dual_mul.real, 6.);
        assert_eq!(dual_mul.dual, 17.);
    }

    fn cube(dual_number: DualNumber) -> DualNumber {
        dual_number.pow(DualNumber::new(3., 0.))
    }

    // Expression: f(x) = 2x^2 + exp(5x)
    //             f'(x)= 4x + 5 * exp(5x)
    fn test_exp_fn(dual_number: DualNumber) -> DualNumber {
        // dual_number * dual_number.pow(DualNumber::new(2., 0.)) +
        dual_number * dual_number * dual_number
            + DualNumber::new(f64::exp(1.), 0.).pow(DualNumber::new(5., 0.) * dual_number)
    }

    fn test_exp_fn_deriv(value: f64) -> f64 {
        4. * value + 5. * f64::exp(1.).powf(5. * value)
    }

    #[test]
    fn test_cube() {
        let dual_number = DualNumber::new(0.1, 1.);
        let result = cube(dual_number);
        assert_approx_eq!(f64, result.real, 0.001);
        assert_approx_eq!(f64, result.dual, 0.03);
    }

    #[test]
    fn test_grad() {
        assert_approx_eq!(f64, DualNumber::grad(cube, 0.1), 0.03);
        assert_approx_eq!(f64, grad(cube, 0.1), 0.03);
        assert_approx_eq!(f64, cube.grad(0.1), 0.03);
        // assert_approx_eq!(f64, test_exp_fn.grad(3.0), 16_345_098.862_360_554_f64);
        assert_approx_eq!(f64, test_exp_fn.grad(3.0), test_exp_fn_deriv(3.0));

        let f = |x: DualNumber| x.pow(DualNumber::new(10., 0.));
        let fp = |x: f64| 10. * x.powf(9.);
        assert_approx_eq!(f64, f.grad(10.), fp(10.));
        let f = |x: DualNumber| DualNumber::new(f64::exp(1.), 0.).pow(DualNumber::new(10., 0.) * x);
        // let fp = |x: f64| 10. * f64::exp(10. * x);
        // assert_approx_eq!(f64, f.grad(2.), fp(2.));
    }
}

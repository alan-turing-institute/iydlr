use std::{
    cell::RefCell,
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
    rc::Rc,
};

use interfaces::{
    tensors::{Element, RealElement},
    utils::{Exp, Ln, Pow},
};
use num_traits::identities::Zero;

const NEG_INF: f64 = -1_000_000.0;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
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
impl Sub for DualNumber {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.real - rhs.real, self.dual - rhs.dual)
    }
}

impl SubAssign for DualNumber {
    fn sub_assign(&mut self, rhs: Self) {
        self.real -= rhs.real;
        self.dual -= rhs.dual;
    }
}

impl Mul for DualNumber {
    type Output = Self;
    //  (x + ie) * (a + be) = (ax + (ay + xb) e)
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
    pub fn grad_fn<F: Fn(DualNumber) -> DualNumber>(func: F, value: f64) -> f64 {
        func(DualNumber::new(value, 1.)).dual
    }
}

pub trait Grad {
    fn grad_fn(&self, value: f64) -> f64;
}

// Blanket implementation for Fn(f64) -> f64
impl<F> Grad for F
where
    F: Fn(DualNumber) -> DualNumber,
{
    fn grad_fn(&self, value: f64) -> f64 {
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

impl DualNumber {
    pub fn val(&self) -> f64 {
        self.real
    }
    pub fn grad(&self) -> f64 {
        self.dual
    }
    pub fn set_val(&mut self, value: f64) {
        self.real = value;
    }
    pub fn set_grad(&mut self, value: f64) {
        self.real = value;
    }
}

impl DualNumberPtr {
    pub fn val(&self) -> f64 {
        (*self.inner).borrow().val()
    }
    pub fn grad(&self) -> f64 {
        (*self.inner).borrow().dual
    }
    pub fn set_val(&mut self, value: f64) {
        (*self.inner).borrow_mut().real = value;
    }
    pub fn set_grad(&mut self, value: f64) {
        (*self.inner).borrow_mut().dual = value;
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct DualNumberPtr {
    pub inner: Rc<RefCell<DualNumber>>,
}

impl DualNumberPtr {
    pub fn new(real: f64, dual: f64) -> Self {
        Self {
            inner: Rc::new(RefCell::new(DualNumber::new(real, dual))),
        }
    }
}

impl Add for DualNumberPtr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            inner: Rc::new(RefCell::new(
                *(*self.inner).borrow() + *(*rhs.inner).borrow(),
            )),
        }
    }
}

impl AddAssign for DualNumberPtr {
    fn add_assign(&mut self, rhs: Self) {
        *(*self.inner).borrow_mut() += *(*rhs.inner).borrow();
    }
}

impl Sub for DualNumberPtr {
    type Output = DualNumberPtr;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            inner: Rc::new(RefCell::new(
                *(*self.inner).borrow() - *(*rhs.inner).borrow(),
            )),
        }
    }
}

impl SubAssign for DualNumberPtr {
    fn sub_assign(&mut self, rhs: Self) {
        *(*self.inner).borrow_mut() -= *(*rhs.inner).borrow();
    }
}

impl Mul for DualNumberPtr {
    type Output = DualNumberPtr;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            inner: Rc::new(RefCell::new(
                *(*self.inner).borrow() * *(*rhs.inner).borrow(),
            )),
        }
    }
}

impl MulAssign for DualNumberPtr {
    fn mul_assign(&mut self, rhs: Self) {
        *(*self.inner).borrow_mut() *= *(*rhs.inner).borrow();
    }
}

impl Div for DualNumberPtr {
    type Output = DualNumberPtr;

    fn div(self, rhs: Self) -> Self::Output {
        Self {
            inner: Rc::new(RefCell::new(
                *(*self.inner).borrow() / *(*rhs.inner).borrow(),
            )),
        }
    }
}

impl DivAssign for DualNumberPtr {
    fn div_assign(&mut self, rhs: Self) {
        *(*self.inner).borrow_mut() /= *(*rhs.inner).borrow();
    }
}

impl Pow for DualNumberPtr {
    fn pow(self, exp: Self) -> Self {
        let exp_inner = *(*exp.inner).borrow();
        let pow_inner = (*self.inner).borrow().pow(exp_inner);
        Self {
            inner: Rc::new(RefCell::new(pow_inner)),
        }
    }
}

impl Ln for DualNumberPtr {
    fn ln(self) -> Self {
        Self {
            inner: Rc::new(RefCell::new((*self.inner).borrow().ln())),
        }
    }
}

impl From<f64> for DualNumberPtr {
    fn from(value: f64) -> Self {
        Self {
            inner: Rc::new(RefCell::new(DualNumber {
                real: value,
                dual: 0.,
            })),
        }
    }
}

impl Exp for DualNumberPtr {
    fn exp(self) -> Self {
        Self {
            inner: Rc::new(RefCell::new((*self.inner).borrow().exp())),
        }
    }
}

impl Zero for DualNumberPtr {
    fn zero() -> Self {
        Self {
            inner: Rc::new(RefCell::new(DualNumber::new(0., 0.))),
        }
    }

    fn is_zero(&self) -> bool {
        self.inner.eq(&(DualNumberPtr::zero().inner))
    }
}

impl Display for DualNumberPtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}, {})",
            (*self.inner).borrow().real,
            (*self.inner).borrow().dual
        )
    }
}

impl Element for DualNumberPtr {}

impl RealElement for DualNumberPtr {
    fn neg_inf() -> Self {
        Self::new(-NEG_INF, 0.)
    }
}

impl From<DualNumberPtr> for f64 {
    fn from(value: DualNumberPtr) -> Self {
        (*value.inner).borrow().real
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

    #[test]
    fn test_cube() {
        let dual_number = DualNumber::new(0.1, 1.);
        let result = cube(dual_number);
        assert_approx_eq!(f64, result.real, 0.001);
        assert_approx_eq!(f64, result.dual, 0.03);
    }

    // Expression: f(x) = 2x^2 + exp(5x)
    //             f'(x)= 4x + 5 * exp(5x)
    fn test_exp_fn(dual_number: DualNumber) -> DualNumber {
        DualNumber::new(2., 0.) * dual_number.pow(DualNumber::new(2.0, 0.))
            + DualNumber::new(f64::exp(1.), 0.).pow(DualNumber::new(5., 0.) * dual_number)
    }

    fn test_exp_fn_deriv(value: f64) -> f64 {
        4. * value + 5. * f64::exp(1.).powf(5. * value)
    }

    #[test]
    fn test_grad() {
        assert_approx_eq!(f64, DualNumber::grad_fn(cube, 0.1), 0.03);
        assert_approx_eq!(f64, grad(cube, 0.1), 0.03);
        assert_approx_eq!(f64, cube.grad_fn(0.1), 0.03);
        assert_approx_eq!(f64, test_exp_fn.grad_fn(3.0), test_exp_fn_deriv(3.0));
        assert_approx_eq!(f64, test_exp_fn.grad_fn(6.0), test_exp_fn_deriv(6.0));
    }
}

use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign},
};

#[derive(Debug, Clone, Copy)]
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
    //  (x + yi) / (a + bi)
    //  = (x + yi)(a - bi) / ((a + bi)(a-bi))
    //  = (ax + (ay - xb)i) / (a**2)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual() {
        let dual_a = DualNumber::new(3., 4.);
        let dual_b = DualNumber::new(2., 3.);
        let dual_mul = dual_a * dual_b;
        assert_eq!(dual_mul.real, 6.);
        assert_eq!(dual_mul.dual, 17.);
    }

    fn cube(dual_number: DualNumber) -> DualNumber {
        dual_number * dual_number * dual_number
    }

    #[test]
    fn test_cube() {
        let dual_number = DualNumber::new(0.1, 1.);
        let result = cube(dual_number);
        println!("{}", result);
    }
}

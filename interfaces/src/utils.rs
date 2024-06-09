/// Raise Euler's number (e ~= 2.71828) to the power of `self`. Eg. `e^self`.
pub trait Exp {
    fn exp(self) -> Self;
}

/// Raise `self` to the power of the exponent `exp`. Eg. `self^exp`. By default the type of the
/// exponent is the same as `self` (a.k.a `Self` capatalised), but this trait can be implemented with
/// a different generic (eg. `Impl Pow<f64> for MyType {}`).
pub trait Pow<Exponent = Self> {
    fn pow(self, exp: Exponent) -> Self;
}

/// Take to natural logarithm of `self`.
pub trait Ln {
    fn ln(self) -> Self;
}

// The below implementations are required for f64 to implement `RealElement`.
impl Exp for f64 {
    fn exp(self) -> Self {
        self.exp()
    }
}

impl Pow for f64 {
    fn pow(self, exp: Self) -> Self {
        self.powf(exp)
    }
}

impl Ln for f64 {
    fn ln(self) -> Self {
        self.ln()
    }
}

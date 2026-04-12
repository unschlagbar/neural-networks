// ── Activation trait ──────────────────────────────────────────────────────────

/// An activation function. Each variant is a zero-sized struct, so the vtable
/// is the only heap cost — and that only applies when stored as `Box<dyn Activate>`.
/// When the concrete type is known at call sites the compiler inlines everything.
pub trait Activate: Send + Sync + std::fmt::Debug + Sized + 'static {
    fn activate(&self, x: &mut [f32]);
    /// Derivative applied to *already-activated* values (in-place).
    fn derivative_active(&self, x: &mut [f32]);

    fn activation_id(&self) -> u8;
}

#[derive(Debug, Clone, Copy)]
pub struct Relu;

#[derive(Debug, Clone, Copy)]
pub struct LeakyRelu;

#[derive(Debug, Clone, Copy)]
pub struct Sigmoid;

#[derive(Debug, Clone, Copy)]
pub struct Tanh;

#[derive(Debug, Clone, Copy)]
pub struct Linear;

impl Activate for Linear {
    #[inline]
    fn activate(&self, _x: &mut [f32]) {}
    #[inline]
    fn derivative_active(&self, x: &mut [f32]) {
        x.iter_mut().for_each(|v| *v = 1.0);
    }
    fn activation_id(&self) -> u8 {
        0
    }
}

impl Activate for Relu {
    #[inline]
    fn activate(&self, x: &mut [f32]) {
        x.iter_mut().for_each(|v| *v = v.max(0.0));
    }
    #[inline]
    fn derivative_active(&self, x: &mut [f32]) {
        x.iter_mut()
            .for_each(|v| *v = if *v > 0.0 { 1.0 } else { 0.0 });
    }
    fn activation_id(&self) -> u8 {
        1
    }
}

impl Activate for LeakyRelu {
    #[inline]
    fn activate(&self, x: &mut [f32]) {
        x.iter_mut().for_each(|v| {
            if *v < 0.0 {
                *v *= 0.01;
            }
        });
    }
    #[inline]
    fn derivative_active(&self, x: &mut [f32]) {
        x.iter_mut()
            .for_each(|v| *v = if *v > 0.0 { 1.0 } else { 0.01 });
    }
    fn activation_id(&self) -> u8 {
        4
    }
}

impl Activate for Tanh {
    #[inline]
    fn activate(&self, x: &mut [f32]) {
        x.iter_mut().for_each(|v| *v = v.tanh());
    }
    fn derivative_active(&self, x: &mut [f32]) {
        x.iter_mut().for_each(|v| *v = 1.0 - *v * *v);
    }
    fn activation_id(&self) -> u8 {
        2
    }
}

impl Activate for Sigmoid {
    #[inline]
    fn activate(&self, x: &mut [f32]) {
        x.iter_mut().for_each(|v| *v = sigmoid(*v));
    }
    #[inline]
    fn derivative_active(&self, x: &mut [f32]) {
        // σ(x) is already stored; σ'(x) = σ(x)·(1 − σ(x))
        x.iter_mut().for_each(|v| *v *= 1.0 - *v);
    }
    fn activation_id(&self) -> u8 {
        3
    }
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

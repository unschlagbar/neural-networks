use iron_oxide::collections::Matrix;
use rand::{Rng, rng};

use crate::{
    lstm::{add_vec_in_place, outer},
};

#[derive(Debug)]
pub struct DenseLayer {
    pub weights: Matrix,
    pub biases: Vec<f32>,
    pub activation: Activation,
}

impl DenseLayer {
    pub fn random(input_size: usize, hidden_size: usize, activation: Activation) -> Self {
        let mut rng = rng();

        let weights = Matrix::random(input_size, hidden_size, 1.0);
        let biases = (0..hidden_size)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        Self {
            weights,
            biases,
            activation,
        }
    }

    pub fn validate_cache(&mut self, cache: &(Vec<f32>, Vec<f32>)) {
        let input_size = self.input_size();
        let hidden_size = self.hidden_size();

        assert_eq!(cache.0.len(), input_size);
        assert_eq!(cache.1.len(), hidden_size);
    }

    pub fn forward(&mut self, input: &[f32], cache: &mut (Vec<f32>, Vec<f32>)) {
        assert_eq!(input.len(), self.weights.rows());
        #[cfg(debug_assertions)]
        self.validate_cache(cache);

        cache.1.copy_from_slice(&self.biases);
        let z = &mut cache.1;
        for (i, &x) in input.iter().enumerate() {
            for (j, &w) in self.weights[i].iter().enumerate() {
                z[j] += x * w;
            }
        }

        cache.0.copy_from_slice(input);

        self.activation.activate(z);
    }

    pub fn forward_no_activation(&mut self, input: &[f32], cache: &mut (Vec<f32>, Vec<f32>)) {
        assert_eq!(input.len(), self.weights.rows());

        cache.1.copy_from_slice(&self.biases);
        let z = &mut cache.1;
        for (i, &x) in input.iter().enumerate() {
            for (j, &w) in self.weights[i].iter().enumerate() {
                z[j] += x * w;
            }
        }
    }

    pub fn backwards(
        &mut self,
        cache: &mut (Vec<f32>, Vec<f32>),
        delta: &mut [f32],
        grads: &mut DenseLayerGrads,
        delta_next: Option<&mut [f32]>,
    ) {
        if !matches!(self.activation, Activation::Softmax) {
            self.activation.derivative_active(&mut cache.1);

            delta.iter_mut().zip(&cache.1).for_each(|(d, o)| *d *= o);
        }

        grads.weights.add_inplace(&outer(&cache.0, delta));
        add_vec_in_place(&mut grads.biases, delta);

        if let Some(delta_next) = delta_next {
            // dh for top layer receives Wy^T * dy plus dhNext from future
            for (i, d) in delta_next.iter_mut().enumerate() {
                let mut s = 0.0;
                for (&dy, &weight) in delta.iter().zip(&self.weights[i]) {
                    s += dy * weight;
                }
                *d += s;
            }
        }
    }

    pub const fn input_size(&self) -> usize {
        self.weights.rows()
    }

    pub const fn hidden_size(&self) -> usize {
        self.weights.cols()
    }
}

pub struct DenseLayerGrads {
    pub weights: Matrix,
    pub biases: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
    Linear,
}

impl Activation {
    pub fn activate(&self, x: &mut [f32]) {
        match self {
            Self::Relu => x.iter_mut().for_each(|x| *x = x.max(0.0)),
            Self::Sigmoid => x.iter_mut().for_each(|x| *x = sigmoid(*x)),
            Self::Tanh => x.iter_mut().for_each(|x| *x = x.tanh()),
            Self::Softmax => x.copy_from_slice(&softmax(x)),
            Self::Linear => (),
        }
    }

    pub fn derivative(&self, x: &mut [f32]) {
        match self {
            Self::Relu => x
                .iter_mut()
                .for_each(|x| *x = if *x > 0.0 { 1.0 } else { 0.0 }),
            Self::Sigmoid => {
                self.activate(x);
                x.iter_mut().for_each(|x| *x = *x * (1.0 - *x));
            }
            Self::Tanh => unimplemented!(),
            Self::Softmax => unimplemented!(),
            Self::Linear => (),
        }
    }

    pub fn derivative_active(&self, x: &mut [f32]) {
        match self {
            Self::Relu => x
                .iter_mut()
                .for_each(|x| *x = if *x == 0.0 { 0.0 } else { 1.0 }),
            Self::Sigmoid => {
                x.iter_mut().for_each(|x| *x = *x * (1.0 - *x));
            }
            Self::Tanh => unimplemented!(),
            Self::Softmax => unimplemented!(),
            Self::Linear => (),
        }
    }
}

pub fn softmax(vec: &[f32]) -> Box<[f32]> {
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0;

    for i in vec {
        max = i.max(max);
    }

    let mut out: Box<[f32]> = vec
        .iter()
        .map(|x| {
            let e = (x - max).exp();
            sum += e;
            e
        })
        .collect();

    out.iter_mut().for_each(|x| *x /= sum);

    out
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub trait Cache {}

pub trait Grads {}


pub trait Layer<C: Cache, G: Grads> {

    fn forward(&self, input: &[f32], cache: &mut C);
    fn backwards(
        &mut self,
        cache: &mut C,
        delta: &mut [f32],
        grads: &mut G,
        delta_next: Option<&mut [f32]>,
    );
}

impl Grads for DenseLayerGrads {}
impl Cache for Vec<f32> {}

impl Layer<Vec<f32>, DenseLayerGrads> for DenseLayer {
    fn forward(&self, input: &[f32], cache: &mut Vec<f32>) {
        todo!()
    }

    fn backwards(
        &mut self,
        cache: &mut Vec<f32>,
        delta: &mut [f32],
        grads: &mut DenseLayerGrads,
        delta_next: Option<&mut [f32]>,
    ) {
        todo!()
    }
}

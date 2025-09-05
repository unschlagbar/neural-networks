use iron_oxide::collections::Matrix;
use rand::{Rng, rng};

use crate::{
    lstm::{add_vec_in_place, outer},
    mlp::DenseLayerGrads,
};

#[derive(Debug)]
pub struct DenseLayer {
    pub weights: Matrix,
    pub biases: Vec<f32>,
    pub activation: Activation,
    pub last_z: Vec<f32>,
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
            last_z: vec![0.0; hidden_size],
        }
    }

    pub fn forward(&mut self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(input.len(), self.weights.rows());
        let last_input = input.to_vec();

        let mut z = self.biases.to_vec();
        for (i, &x) in input.iter().enumerate() {
            for (j, &w) in self.weights[i].iter().enumerate() {
                z[j] += x * w;
            }
        }
        self.last_z.copy_from_slice(&z);

        self.activation.activate(&mut z);
        (last_input, z)
    }

    pub fn forward_no_activation(&mut self, input: &[f32]) -> &[f32] {
        assert_eq!(input.len(), self.weights.rows());
        //self.last_input.copy_from_slice(input);

        let mut z = self.biases.clone();
        for (i, &x) in input.iter().enumerate() {
            for (j, &w) in self.weights[i].iter().enumerate() {
                z[j] += x * w;
            }
        }
        self.last_z = z;
        &self.last_z
    }

    pub fn backwards(
        &mut self,
        input: &[f32],
        delta: &[f32],
        grads: &mut DenseLayerGrads,
        delta_next: Option<&mut [f32]>,
    ) {
        grads.weights.add_inplace(&outer(input, delta));
        add_vec_in_place(&mut grads.biases, delta);

        if let Some(delta_next) = delta_next {
            // dh for top layer receives Wy^T * dy plus dhNext from future
            for i in 0..self.input_size() {
                let mut s = 0.0;
                for (&dy, &weight) in delta.iter().zip(&self.weights[i]) {
                    s += dy * weight;
                }
                delta_next[i] += s;
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

    pub fn derivative_active(&self, x: Box<[f32]>) -> Box<[f32]> {
        match self {
            Self::Relu => x
                .into_iter()
                .map(|x| if x > 0.0 { 1.0 } else { 0.0 })
                .collect(),
            Self::Sigmoid => x.into_iter().map(|x| x * (1.0 - x)).collect(),
            Self::Tanh => unimplemented!(),
            Self::Softmax => unimplemented!(),
            Self::Linear => x,
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

use iron_oxide::collections::Matrix;
use rand::{Rng, rng};

#[derive(Debug)]
pub struct LearningLayer {
    pub weights: Matrix,
    pub biases: Box<[f32]>,
    pub activation_fn: Activation,
    pub last_input: Box<[f32]>,
    pub last_z: Box<[f32]>,
}

impl LearningLayer {
    pub fn random(inputs: usize, size: usize, activation_fn: Activation) -> Self {
        let mut rng = rng();

        let weights: Vec<f32> = (0..inputs * size)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        let weights = Matrix::from_vec(weights, inputs, size);

        let biases = (0..size).map(|_| rng.random_range(-1.0..1.0)).collect();

        Self {
            weights,
            biases,
            activation_fn,
            last_input: vec![0.0; inputs].into_boxed_slice(),
            last_z: vec![0.0; size].into_boxed_slice(),
        }
    }

    pub fn forward(&mut self, input: &[f32]) -> Box<[f32]> {
        assert_eq!(input.len(), self.weights.rows());
        self.last_input.copy_from_slice(input);

        let mut z = self.biases.clone();
        for (i, &x) in input.iter().enumerate() {
            for (j, &w) in self.weights[i].iter().enumerate() {
                z[j] += x * w;
            }
        }
        self.last_z.copy_from_slice(&z);

        self.activation_fn.activate(&mut z);
        z
    }

    pub fn forward_no_activation(&mut self, input: &[f32]) -> Box<[f32]> {
        assert_eq!(input.len(), self.weights.rows());
        self.last_input.copy_from_slice(input);

        let mut z = self.biases.clone();
        for (i, &x) in input.iter().enumerate() {
            for (j, &w) in self.weights[i].iter().enumerate() {
                z[j] += x * w;
            }
        }
        self.last_z.copy_from_slice(&z);
        z
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

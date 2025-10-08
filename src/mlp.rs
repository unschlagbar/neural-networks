use std::f32::consts::PI;

use iron_oxide::collections::Matrix;
use rand::{Rng, rng, seq::SliceRandom};

use crate::layer::{Activation, DenseLayer};

#[derive(Debug)]
pub struct LearningMLP {
    pub layers: Box<[DenseLayer]>,
}

impl LearningMLP {
    pub fn random(layout: &[usize]) -> Self {
        let mut layers: Box<[DenseLayer]> = layout
            .windows(2)
            .map(|window| DenseLayer::random(window[0], window[1], Activation::Relu))
            .collect();

        layers[layers.len() - 1].activation = Activation::Sigmoid;

        Self { layers }
    }

    pub fn forward(&mut self, _input: &[f32]) -> Vec<f32> {
        //let gg = (vec![0.0; input.len()], vec![0.0; self])
        //let mut input = self.layers[0].forward(input).1;
        //for layer in &mut self.layers[1..] {
        //    input = layer.forward(&input).1;
        //}
        //input
        todo!()
    }

    pub fn train(&mut self, input: &[f32], target: &[f32], learning_rate: f32) {
        let output = self.forward(input);
        assert_eq!(output.len(), target.len());

        let n_layers = self.layers.len();
        let output_layer_idx = n_layers - 1;

        let output_fn = self.layers[output_layer_idx].activation;
        let mut output_z = self.layers[output_layer_idx].last_z.clone();
        output_fn.derivative_active(&mut output_z);

        let mut delta: Box<[f32]> = (0..output.len())
            .map(|j| (output[j] - target[j]) * output_z[j])
            .collect();

        for l in (0..n_layers).rev() {
            // fix
            //let curr_input: &[f32] = &self.layers[l].last_input;
            let curr_input: &[f32] = &vec![0.0; self.layers[l].input_size()];

            // Update weights
            for (i, input) in curr_input.iter().enumerate() {
                for (j, delta) in delta.iter().enumerate() {
                    self.layers[l].weights[i][j] -= learning_rate * input * delta;
                }
            }

            // Update biases
            for (delta, bias) in delta.iter().zip(self.layers[l].biases.iter_mut()) {
                *bias -= learning_rate * delta;
            }

            // If not the first layer, compute delta for the previous layer
            if l > 0 {
                let prev_idx = l - 1;
                let prev_fn = self.layers[prev_idx].activation;
                let mut prev_z = self.layers[prev_idx].last_z.clone();
                let prev_size = self.layers[l].weights.rows();
                let prev_delta: Box<[f32]> = (0..prev_size)
                    .map(|k| {
                        let mut sum = 0.0;
                        for (j, delta) in delta.iter().enumerate() {
                            sum += self.layers[l].weights[k][j] * delta;
                        }
                        sum
                    })
                    .collect();
                prev_fn.derivative_active(&mut prev_z);

                delta = prev_delta
                    .into_iter()
                    .zip(&prev_z)
                    .map(|(p, d)| p * d)
                    .collect();
            }
        }
    }

    #[allow(unused)]
    pub fn test() {
        let mut rng = rng();

        let n_points_per_class = 200;
        let noise = 0.3;
        let num_turns = 2.5;
        let max_theta = num_turns * 2.0 * PI;

        let mut inputs: Vec<Vec<f32>> = Vec::new();
        let mut targets: Vec<Vec<f32>> = Vec::new();

        for c in 0..2 {
            for _ in 0..n_points_per_class {
                let rand_val = rng.random::<f32>();
                let theta = rand_val.sqrt() * max_theta;

                let mut x = theta * theta.cos();
                let mut y = theta * theta.sin();

                if c == 1 {
                    x = -x;
                    y = -y;
                }

                x += rng.random_range(-noise..noise);
                y += rng.random_range(-noise..noise);

                inputs.push(vec![x, y]);
                targets.push(vec![c as f32]);
            }
        }

        let mut mlp = LearningMLP::random(&[2, 128, 1]);

        let learning_rate = 0.0051;
        let epochs = 50000;

        for epoch in 0..epochs {
            let mut indices: Vec<usize> = (0..inputs.len()).collect();
            indices.shuffle(&mut rng);

            for &i in &indices {
                mlp.train(&inputs[i], &targets[i], learning_rate);
            }

            let mut total_loss = 0.0;
            let mut correct = 0;

            for i in 0..inputs.len() {
                let output = mlp.forward(&inputs[i]);
                let loss = (output[0] - targets[i][0]).powi(2);
                total_loss += loss;

                if (output[0] > 0.5) == (targets[i][0] > 0.5) {
                    correct += 1;
                }
            }

            let avg_loss = total_loss / inputs.len() as f32;
            let acc = correct as f32 / inputs.len() as f32 * 100.0;

            if epoch % 1000 == 0 || epoch == epochs - 1 {
                println!("Epoch {epoch}: Average Loss = {avg_loss:.3}, Accuracy = {acc:.2}%");
            }
        }

        println!("\nSample test results (first 10):");
        for i in 0..20 {
            let i = if i % 2 == 0 { i } else { i + 200 };
            let output = mlp.forward(&inputs[i]);
            println!(
                "{:?} -> {:.3}, Expected: {:.3}",
                inputs[i], output[0], targets[i][0]
            );
        }
    }
}

pub struct DenseLayerGrads {
    pub weights: Matrix,
    pub biases: Vec<f32>,
}

mod adam;
mod sgd;

pub use adam::*;
use ndarray::{ArrayViewD, ArrayViewMutD};
pub use sgd::*;

use crate::{CPULayer, Optimizer};

pub enum CPUOptimizer {
    SGD(CPUSGDOptimizer),
    Adam(CPUAdamOptimizer),
}

impl CPUOptimizer {
    pub fn from(optimizer: Optimizer, layers: &mut Vec<CPULayer>) -> Self {
        let mut all_params = Vec::new();
        for layer in layers {
            if let Some((params, _)) = CPUOptimizer::get_params(layer) {
                all_params.push(params)
            }
        }
        match optimizer {
            Optimizer::SGD => CPUOptimizer::SGD(CPUSGDOptimizer::new()),
            Optimizer::Adam(config) => {
                CPUOptimizer::Adam(CPUAdamOptimizer::new(config, all_params))
            }
        }
    }

    pub fn update_grads(&mut self, layers: &mut Vec<CPULayer>, rate: f32) {
        match self {
            CPUOptimizer::Adam(adam) => adam.t += 1.0,
            _ => {}
        }
        let mut idx = 0;
        for layer in layers.iter_mut() {
            if let Some((params, grads)) = CPUOptimizer::get_params(layer) {
                match self {
                    CPUOptimizer::SGD(sgd) => sgd.update_grads(params, grads, rate),
                    CPUOptimizer::Adam(adam) => adam.update_grads(params, grads, idx, rate),
                }
                idx += 1;
            }
        }
    }

    pub fn get_params<'a>(
        layer: &'a mut CPULayer,
    ) -> Option<(Vec<ArrayViewMutD<'a, f32>>, Vec<ArrayViewD<'a, f32>>)> {
        match layer {
            CPULayer::Dense(layer) => Some((
                vec![
                    layer.weights.view_mut().into_dyn(),
                    layer.biases.view_mut().into_dyn(),
                ],
                vec![
                    layer.d_weights.view().into_dyn(),
                    layer.d_biases.view().into_dyn(),
                ],
            )),
            CPULayer::Conv2D(layer) => Some((
                vec![
                    layer.weights.view_mut().into_dyn(),
                    layer.biases.view_mut().into_dyn(),
                ],
                vec![
                    layer.d_weights.view().into_dyn(),
                    layer.d_biases.view().into_dyn(),
                ],
            )),
            CPULayer::ConvTranspose2D(layer) => Some((
                vec![
                    layer.weights.view_mut().into_dyn(),
                    layer.biases.view_mut().into_dyn(),
                ],
                vec![
                    layer.d_weights.view().into_dyn(),
                    layer.d_biases.view().into_dyn(),
                ],
            )),
            CPULayer::BatchNorm1D(layer) => Some((
                vec![
                    layer.gamma.view_mut().into_dyn(),
                    layer.beta.view_mut().into_dyn(),
                ],
                vec![
                    layer.d_gamma.view().into_dyn(),
                    layer.d_beta.view().into_dyn(),
                ],
            )),
            CPULayer::BatchNorm2D(layer) => Some((
                vec![
                    layer.gamma.view_mut().into_dyn(),
                    layer.beta.view_mut().into_dyn(),
                ],
                vec![
                    layer.d_gamma.view().into_dyn(),
                    layer.d_beta.view().into_dyn(),
                ],
            )),
            _ => return None,
        }
    }
}

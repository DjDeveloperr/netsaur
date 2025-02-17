use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BackendConfig {
    pub silent: Option<bool>,
    pub size: Vec<usize>,
    pub layers: Vec<Layer>,
    pub cost: Cost,
    pub optimizer: Optimizer,
}

#[derive(Debug)]
pub struct Dataset {
    pub inputs: ArrayD<f32>,
    pub outputs: ArrayD<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "config")]
#[serde(rename_all = "lowercase")]
pub enum Layer {
    Activation(ActivationLayer),
    Dense(DenseLayer),
    BatchNorm1D(BatchNormLayer),
    BatchNorm2D(BatchNormLayer),
    Conv2D(Conv2DLayer),
    ConvTranspose2D(ConvTranspose2DLayer),
    Pool2D(Pool2DLayer),
    Flatten(FlattenLayer),
    Dropout1D(DropoutLayer),
    Dropout2D(DropoutLayer),
    Softmax,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    Elu,
    LeakyRelu,
    Linear,
    Relu,
    Relu6,
    Selu,
    Sigmoid,
    Tanh,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct JSTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DenseLayer {
    pub size: Vec<usize>,
    pub init: Option<Init>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Conv2DLayer {
    pub init: Option<Init>,
    pub kernel: Option<JSTensor>,
    pub kernel_size: Vec<usize>,
    pub padding: Option<Vec<usize>>,
    pub strides: Option<Vec<usize>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ConvTranspose2DLayer {
    pub init: Option<Init>,
    pub kernel: Option<JSTensor>,
    pub kernel_size: Vec<usize>,
    pub padding: Option<Vec<usize>>,
    pub strides: Option<Vec<usize>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Pool2DLayer {
    pub mode: usize, // 0 = avg, 1 = max
    pub strides: Option<Vec<usize>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FlattenLayer {
    pub size: Vec<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DropoutLayer {
    pub probability: f32,
    pub inplace: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BatchNormLayer {
    pub momentum: f32,
    pub epsilon: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ActivationLayer {
    pub activation: Activation,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Cost {
    CrossEntropy,
    Hinge,
    MSE,
    BinCrossEntropy,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Init {
    Uniform,
    Xavier,
    XavierN,
    Kaiming,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub struct  AdamOptimizer {
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "config")]
#[serde(rename_all = "lowercase")]
pub enum Optimizer {
    SGD,
    Adam(AdamOptimizer)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TrainOptions {
    pub datasets: usize,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub epochs: usize,
    pub batches: usize,
    pub rate: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PredictOptions {
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}

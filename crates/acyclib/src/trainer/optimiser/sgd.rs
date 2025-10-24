use std::{collections::HashMap, sync::Arc};

use crate::device::{Device, OperationError, operation::BaseOperations, tensor::DenseMatrix};

use super::{OptimiserState, utils};

#[derive(Clone, Copy, Debug)]
pub struct SgdParams {
    pub decay: f32,
    pub momentum: f32,
    pub nesterov: bool,
    pub min_weight: f32,
    pub max_weight: f32,
}

impl Default for SgdParams {
    fn default() -> Self {
        Self { decay: 0.0, momentum: 0.0, nesterov: false, min_weight: -1.98, max_weight: 1.98 }
    }
}

pub struct SGD<D: Device> {
    velocity: DenseMatrix<D>,
    params: SgdParams,
}

impl<D: Device> OptimiserState<D> for SGD<D> {
    type Params = SgdParams;

    fn new(device: Arc<D>, size: usize, params: Self::Params) -> Result<Self, D::DeviceError> {
        if params.max_weight < params.min_weight {
            return Err(D::DeviceError::default());
        }

        Ok(Self { velocity: DenseMatrix::zeroed(device, size, None)?, params })
    }

    fn update(
        &mut self,
        weights: &mut DenseMatrix<D>,
        grads: &mut DenseMatrix<D>,
        gradient_factor: f32,
        learning_rate: f32,
    ) -> Result<(), OperationError<D::DeviceError>> {
        assert!(weights.batch_size().is_none());
        assert!(self.velocity.batch_size().is_none());
        assert_eq!(weights.size(), self.velocity.size());

        let size = weights.size();
        let momentum = self.params.momentum;
        let use_momentum = momentum.abs() > f32::EPSILON;

        grads.buf.mul_scalar(size, gradient_factor).map_err(OperationError::from)?;

        if self.params.decay != 0.0 {
            grads.buf.linear_comb(size, 1.0, self.params.decay, &weights.buf).map_err(OperationError::from)?;
        }

        if use_momentum {
            self.velocity.buf.mul_scalar(size, momentum).map_err(OperationError::from)?;
            self.velocity.buf.linear_comb(size, 1.0, 1.0, &grads.buf).map_err(OperationError::from)?;
            if self.params.nesterov {
                grads.buf.linear_comb(size, 1.0, momentum, &self.velocity.buf).map_err(OperationError::from)?;
            } else {
                grads.copy_from(&self.velocity).map_err(OperationError::from)?;
            }
        }

        weights.buf.linear_comb(size, 1.0, -learning_rate, &grads.buf).map_err(OperationError::from)?;

        if self.params.min_weight < self.params.max_weight {
            weights.buf.clip(size, self.params.min_weight, self.params.max_weight).map_err(OperationError::from)?;
        }

        Ok(())
    }

    fn reset(&mut self) -> Result<(), D::DeviceError> {
        self.velocity.set_to(0.0)
    }

    fn write_to_checkpoint(map: &HashMap<String, &Self>, path: &str) -> Result<(), D::DeviceError> {
        let velocity: Vec<_> = map.iter().map(|(id, single)| (id, &single.velocity)).collect();
        utils::write_weights_to_file(&velocity, &format!("{path}/velocity.bin"))
    }

    fn load_from_checkpoint(
        map: &mut HashMap<String, &mut Self>,
        path: &str,
        old_format: bool,
    ) -> Result<(), OperationError<D::DeviceError>> {
        let mut velocity = utils::load_weights_from_file(&format!("{path}/velocity.bin"), old_format);
        velocity.sort_by_key(|(id, _)| id.clone());

        for (id, values) in velocity {
            if let Some(single) = map.get_mut(&id) {
                single.velocity.load_from_slice(None, &values)?;
            }
        }

        Ok(())
    }

    fn set_params(&mut self, params: Self::Params) {
        assert!(params.max_weight >= params.min_weight, "SGD max_weight must be >= min_weight");
        self.params = params;
    }
}

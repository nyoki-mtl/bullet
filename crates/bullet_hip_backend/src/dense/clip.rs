use crate::{
    DeviceError,
    backend::{Buffer, ops},
};

pub fn clip_assign(
    size: usize,
    input: &Buffer<f32>,
    output: &mut Buffer<f32>,
    min: f32,
    max: f32,
) -> Result<(), DeviceError> {
    if size > input.size() || size > output.size() {
        return Err(DeviceError::ExpectedIllegalAddressAccess);
    }

    unsafe {
        ops::clip_forward(size, input.ptr(), output.mut_ptr(), min, max);
    }

    Ok(())
}

pub fn clip_backward(
    size: usize,
    input: &Buffer<f32>,
    output_grad: &Buffer<f32>,
    input_grad: &mut Buffer<f32>,
    min: f32,
    max: f32,
) -> Result<(), DeviceError> {
    if size > input.size() || size > output_grad.size() || size > input_grad.size() {
        return Err(DeviceError::ExpectedIllegalAddressAccess);
    }

    unsafe {
        ops::clip_backward(size, input.ptr(), output_grad.ptr(), input_grad.mut_ptr(), min, max);
    }

    Ok(())
}

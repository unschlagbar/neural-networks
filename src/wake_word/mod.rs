pub mod buffer;
pub mod detector;
pub mod mfcc;
pub mod model;
pub mod record;
pub mod training;

use cpal::traits::DeviceTrait;

/// Try to open an input config at exactly WAKE_SR (16 kHz).
/// Falls back to the device default if 16 kHz is not supported.
pub(crate) fn preferred_input_config(device: &cpal::Device) -> cpal::SupportedStreamConfig {
    let target = crate::config::WAKE_SR as u32;
    if let Ok(configs) = device.supported_input_configs() {
        for range in configs {
            let fmt = range.sample_format();
            if (fmt == cpal::SampleFormat::F32 || fmt == cpal::SampleFormat::I16)
                && range.min_sample_rate() <= target
                && target <= range.max_sample_rate()
            {
                return range.with_sample_rate(target);
            }
        }
    }
    eprintln!("16 kHz not available on this device — falling back to default (will resample)");
    device.default_input_config().expect("no input config")
}

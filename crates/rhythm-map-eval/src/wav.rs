use std::{
    fs::File,
    io::{self, Write},
    path::Path,
};

use crate::{GeneratedTruth, SyntheticRecipe};

pub(crate) fn render_click_track(
    recipe: &SyntheticRecipe,
    truth: &GeneratedTruth,
    path: &Path,
) -> io::Result<()> {
    let sample_rate = recipe.sample_rate;
    let sample_count_f64 = (truth.duration_s * f64::from(sample_rate)).ceil();
    if sample_count_f64 > f64::from(u32::MAX / 2) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "rendered WAV would exceed the RIFF size limit",
        ));
    }
    let sample_count = bounded_sample_index(sample_count_f64);
    let mut samples = vec![0.0_f32; sample_count];
    for beat in &truth.beats {
        let frequency = if beat.downbeat { 1_760.0 } else { 1_100.0 };
        let length = bounded_sample_index(0.035 * f64::from(sample_rate));
        let start = bounded_sample_index((beat.time_s * f64::from(sample_rate)).round());
        for offset in 0..length.min(sample_count.saturating_sub(start)) {
            let time = index_to_f64(offset) / f64::from(sample_rate);
            let envelope = (-80.0 * time).exp();
            let value = (std::f64::consts::TAU * frequency * time).sin() * envelope * 0.8;
            samples[start + offset] += f64_to_f32(value);
        }
    }
    write_pcm16_mono(path, sample_rate, &samples)
}

fn write_pcm16_mono(path: &Path, sample_rate: u32, samples: &[f32]) -> io::Result<()> {
    let data_size = u32::try_from(samples.len().saturating_mul(2))
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "WAV exceeds 4 GiB"))?;
    let mut file = File::create(path)?;
    file.write_all(b"RIFF")?;
    file.write_all(&(36_u32 + data_size).to_le_bytes())?;
    file.write_all(b"WAVEfmt ")?;
    file.write_all(&16_u32.to_le_bytes())?;
    file.write_all(&1_u16.to_le_bytes())?;
    file.write_all(&1_u16.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&(sample_rate * 2).to_le_bytes())?;
    file.write_all(&2_u16.to_le_bytes())?;
    file.write_all(&16_u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;
    for sample in samples {
        let quantized = quantize_pcm16(*sample);
        file.write_all(&quantized.to_le_bytes())?;
    }
    Ok(())
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn bounded_sample_index(value: f64) -> usize {
    debug_assert!(value.is_finite() && value >= 0.0 && value <= f64::from(u32::MAX / 2));
    value as usize
}

#[allow(clippy::cast_precision_loss)]
fn index_to_f64(value: usize) -> f64 {
    value as f64
}

#[allow(clippy::cast_possible_truncation)]
fn f64_to_f32(value: f64) -> f32 {
    value as f32
}

#[allow(clippy::cast_possible_truncation)]
fn quantize_pcm16(value: f32) -> i16 {
    (value.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16
}

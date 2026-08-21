use std::{
    fs::File,
    io::{self, Write},
    path::Path,
};

use crate::{GeneratedTruth, SyntheticAudioProfile, SyntheticRecipe};

pub(crate) fn render_synthetic_audio(
    recipe: &SyntheticRecipe,
    truth: &GeneratedTruth,
    path: &Path,
) -> io::Result<()> {
    let samples = synthesize_audio(recipe, truth)?;
    write_pcm16_mono(path, recipe.sample_rate, &samples)
}

pub(crate) fn synthesize_audio(
    recipe: &SyntheticRecipe,
    truth: &GeneratedTruth,
) -> io::Result<Vec<f32>> {
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
    match recipe.audio_profile {
        SyntheticAudioProfile::Click => render_clicks(&mut samples, sample_rate, truth),
        SyntheticAudioProfile::Percussion => render_percussion(&mut samples, sample_rate, truth),
        SyntheticAudioProfile::Drumless => render_drumless(&mut samples, sample_rate, truth),
    }
    Ok(samples)
}

fn render_clicks(samples: &mut [f32], sample_rate: u32, truth: &GeneratedTruth) {
    for beat in &truth.beats {
        add_tone(
            samples,
            sample_rate,
            beat.time_s,
            if beat.downbeat { 1_760.0 } else { 1_100.0 },
            0.035,
            0.8,
            80.0,
        );
    }
}

fn render_percussion(samples: &mut [f32], sample_rate: u32, truth: &GeneratedTruth) {
    for (index, beat) in truth.beats.iter().enumerate() {
        add_tone(
            samples,
            sample_rate,
            beat.time_s,
            if beat.downbeat { 62.0 } else { 78.0 },
            0.18,
            if beat.downbeat { 0.9 } else { 0.65 },
            18.0,
        );
        if index % 4 == 1 || index % 4 == 3 {
            add_noise(samples, sample_rate, beat.time_s, 0.12, 0.35, index as u64);
        }
        add_tone(
            samples,
            sample_rate,
            beat.time_s,
            220.0
                * 2.0_f64.powf(
                    f64::from(u32::try_from(index % 8).expect("modulo result fits u32")) / 12.0,
                ),
            0.22,
            0.16,
            12.0,
        );
        if let Some(next) = truth.beats.get(index + 1)
            && next.time_s - beat.time_s < 1.5
        {
            let subdivision = f64::midpoint(beat.time_s, next.time_s);
            add_noise(
                samples,
                sample_rate,
                subdivision,
                0.035,
                0.12,
                index as u64 + 10_000,
            );
        }
    }
}

fn render_drumless(samples: &mut [f32], sample_rate: u32, truth: &GeneratedTruth) {
    const SCALE: [f64; 8] = [220.0, 246.94, 261.63, 293.66, 329.63, 349.23, 392.0, 440.0];
    for (index, beat) in truth.beats.iter().enumerate() {
        let root = SCALE[index % SCALE.len()];
        let amplitude = if beat.downbeat { 0.32 } else { 0.22 };
        for (ratio, level) in [(1.0, 1.0), (1.25, 0.5), (1.5, 0.35)] {
            add_tone(
                samples,
                sample_rate,
                beat.time_s,
                root * ratio,
                0.32,
                amplitude * level,
                7.0,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn add_tone(
    samples: &mut [f32],
    sample_rate: u32,
    start_s: f64,
    frequency: f64,
    duration_s: f64,
    amplitude: f64,
    decay: f64,
) {
    let start = bounded_sample_index((start_s * f64::from(sample_rate)).round());
    let length = bounded_sample_index(duration_s * f64::from(sample_rate));
    for offset in 0..length.min(samples.len().saturating_sub(start)) {
        let time = index_to_f64(offset) / f64::from(sample_rate);
        let envelope = (-decay * time).exp();
        let value = (std::f64::consts::TAU * frequency * time).sin() * envelope * amplitude;
        samples[start + offset] += f64_to_f32(value);
    }
}

fn add_noise(
    samples: &mut [f32],
    sample_rate: u32,
    start_s: f64,
    duration_s: f64,
    amplitude: f64,
    seed: u64,
) {
    let start = bounded_sample_index((start_s * f64::from(sample_rate)).round());
    let length = bounded_sample_index(duration_s * f64::from(sample_rate));
    let mut state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    for offset in 0..length.min(samples.len().saturating_sub(start)) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let high_bits = u16::try_from(state >> 48).expect("shifted value fits u16");
        let unit = f64::from(high_bits) / f64::from(u16::MAX);
        let time = index_to_f64(offset) / f64::from(sample_rate);
        let value = (unit * 2.0 - 1.0) * (-45.0 * time).exp() * amplitude;
        samples[start + offset] += f64_to_f32(value);
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RecipeSegment, SegmentShape, generate_truth};

    fn recipe(audio_profile: SyntheticAudioProfile) -> SyntheticRecipe {
        SyntheticRecipe {
            schema_version: 1,
            id: "audio-profile-test".to_string(),
            sample_rate: 22_050,
            beats_per_bar: 4,
            audio_profile,
            segments: vec![RecipeSegment {
                duration_s: 2.0,
                shape: SegmentShape::Constant { bpm: 120.0 },
            }],
        }
    }

    #[test]
    fn percussion_render_is_deterministic() {
        let recipe = recipe(SyntheticAudioProfile::Percussion);
        let truth = generate_truth(&recipe).unwrap();
        let first = synthesize_audio(&recipe, &truth).unwrap();
        let second = synthesize_audio(&recipe, &truth).unwrap();
        assert_eq!(first, second);
        assert!(first.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn profiles_render_distinct_audio() {
        let click = recipe(SyntheticAudioProfile::Click);
        let percussion = recipe(SyntheticAudioProfile::Percussion);
        let drumless = recipe(SyntheticAudioProfile::Drumless);
        let truth = generate_truth(&click).unwrap();

        assert_ne!(
            synthesize_audio(&click, &truth).unwrap(),
            synthesize_audio(&percussion, &truth).unwrap()
        );
        assert_ne!(
            synthesize_audio(&percussion, &truth).unwrap(),
            synthesize_audio(&drumless, &truth).unwrap()
        );
    }
}

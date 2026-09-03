//! File decoding and time-aligned preprocessing shared by file and PCM callers.
//! The Symphonia decode scaffold and sinc parameters are adapted from
//! beat-this-rs 1.0.0; see licenses/beat-this-rs-MIT.txt at the repository root.

use std::{borrow::Cow, fs::File, path::Path};

use rhythm_map_core::BackendError;
use rubato::{
    Async, FixedAsync, Indexing, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction, audioadapter_buffers::direct::InterleavedSlice,
};
use symphonia::core::{
    codecs::audio::AudioDecoderOptions,
    formats::{FormatOptions, TrackType, probe::Hint},
    io::{MediaSourceStream, MediaSourceStreamOptions},
    meta::MetadataOptions,
};

use crate::DecodedAudio;

pub(crate) const MODEL_SAMPLE_RATE: u32 = 22_050;
const CHUNK_FRAMES: usize = 4096;

fn audio_error(error: impl std::fmt::Display) -> BackendError {
    BackendError::new(format!("audio preprocessing failed: {error}"))
}

pub(crate) fn decode(path: &Path) -> Result<DecodedAudio, BackendError> {
    let native = decode_native(path)?;
    let samples = if native.sample_rate == MODEL_SAMPLE_RATE {
        native.samples
    } else {
        prepare_mono(&native.samples, native.sample_rate)?.into_owned()
    };
    Ok(DecodedAudio {
        samples,
        sample_rate: MODEL_SAMPLE_RATE,
    })
}

pub(crate) fn decode_native(path: &Path) -> Result<DecodedAudio, BackendError> {
    let source = File::open(path).map_err(audio_error)?;
    let stream = MediaSourceStream::new(Box::new(source), MediaSourceStreamOptions::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
        hint.with_extension(extension);
    }
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(audio_error)?;
    let track = format
        .first_track_known_codec(TrackType::Audio)
        .ok_or_else(|| audio_error("no supported audio track"))?;
    let track_id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or_else(|| audio_error("missing audio codec parameters"))?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .map_err(audio_error)?;
    let mut mono = Vec::new();
    let mut packet_samples = Vec::<f32>::new();
    let mut source_format = None;
    while let Some(packet) = format.next_packet().map_err(audio_error)? {
        if packet.track_id != track_id {
            continue;
        }
        // Dropping corrupt packets would silently compress the musical timeline.
        let packet_audio = decoder.decode(&packet).map_err(audio_error)?;
        let spec = packet_audio.spec();
        let current = (spec.rate(), spec.channels().count());
        if current.0 == 0 || current.1 == 0 || source_format.is_some_and(|old| old != current) {
            return Err(audio_error("invalid or changing audio rate/channel count"));
        }
        source_format = Some(current);
        packet_audio.copy_to_vec_interleaved(&mut packet_samples);
        append_mono(&mut mono, &packet_samples, current.1)?;
    }
    let (rate, _) = source_format.ok_or_else(|| audio_error("no decoded audio packets"))?;
    validate_pcm(&mono, rate)?;
    Ok(DecodedAudio {
        samples: mono,
        sample_rate: rate,
    })
}

fn append_mono(
    output: &mut Vec<f32>,
    interleaved: &[f32],
    channels: usize,
) -> Result<(), BackendError> {
    if channels == 0 || !interleaved.len().is_multiple_of(channels) {
        return Err(audio_error("incomplete interleaved audio frame"));
    }
    if interleaved.iter().any(|sample| !sample.is_finite()) {
        return Err(audio_error("PCM must be finite"));
    }
    // f64 accumulation also avoids overflow when finite float channels are large.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    for frame in interleaved.chunks_exact(channels) {
        output.push(
            (frame.iter().map(|&sample| f64::from(sample)).sum::<f64>() / channels as f64) as f32,
        );
    }
    Ok(())
}

fn validate_pcm(samples: &[f32], rate: u32) -> Result<(), BackendError> {
    if rate == 0 || samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        return Err(audio_error(
            "expected finite nonempty mono PCM and a nonzero sample rate",
        ));
    }
    Ok(())
}

pub(crate) fn prepare_mono(
    samples: &[f32],
    source_rate: u32,
) -> Result<Cow<'_, [f32]>, BackendError> {
    validate_pcm(samples, source_rate)?;
    if source_rate == MODEL_SAMPLE_RATE {
        return Ok(Cow::Borrowed(samples));
    }
    resample(samples, source_rate, CHUNK_FRAMES).map(Cow::Owned)
}

fn resample(
    samples: &[f32],
    source_rate: u32,
    chunk_frames: usize,
) -> Result<Vec<f32>, BackendError> {
    // Integer, nearest-frame duration (ties upward), independent of chunk size.
    let wanted = usize::try_from(
        (samples.len() as u128 * u128::from(MODEL_SAMPLE_RATE) + u128::from(source_rate) / 2)
            / u128::from(source_rate),
    )
    .map_err(audio_error)?;
    if wanted == 0 {
        return Err(audio_error("audio is shorter than one model-rate sample"));
    }
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = Async::<f32>::new_sinc(
        f64::from(MODEL_SAMPLE_RATE) / f64::from(source_rate),
        1.0,
        &params,
        chunk_frames,
        1,
        FixedAsync::Input,
    )
    .map_err(audio_error)?;
    let input = InterleavedSlice::new(samples, 1, samples.len()).map_err(audio_error)?;
    let mut scratch = vec![0.0; resampler.output_frames_max()];
    let mut output = Vec::with_capacity(wanted);
    let mut offset = 0;
    // rubato 3.0's InnerSinc advances by one output period BEFORE evaluating
    // its first sample. output_delay() truncates L/2 * ratio and omits this
    // advance. Use the nearest phase-aligned integer: round(L/2 * ratio) - 1.
    // Residual fractional delay is at most half an output sample, plus sinc
    // table phase quantization. Multi-rate impulses guard dependency upgrades.
    let half_filter = params.sinc_len / 2;
    let mut delay = ((half_filter * MODEL_SAMPLE_RATE as usize + source_rate as usize / 2)
        / source_rate as usize)
        .saturating_sub(1);
    while output.len() < wanted {
        let valid = resampler.input_frames_next().min(samples.len() - offset);
        let indexing = Indexing {
            input_offset: offset,
            output_offset: 0,
            partial_len: Some(valid),
            active_channels_mask: None,
        };
        let scratch_len = scratch.len();
        let mut buffer =
            InterleavedSlice::new_mut(&mut scratch, 1, scratch_len).map_err(audio_error)?;
        let (_, produced) = resampler
            .process_into_buffer(&input, &mut buffer, Some(&indexing))
            .map_err(audio_error)?;
        offset += valid;
        let skip = delay.min(produced);
        delay -= skip;
        let keep = (produced - skip).min(wanted - output.len());
        output.extend_from_slice(&scratch[skip..skip + keep]);
        // After EOF, valid == 0: zero-pad until the delayed tail has emerged.
        // Never append padding to the returned duration or shift model events.
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peak(samples: &[f32]) -> usize {
        samples
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap()
            .0
    }

    #[test]
    fn native_rate_is_borrowed_and_bit_exact() {
        let samples = [0.0, -0.0, 0.25, -0.8, 1.0];
        let output = prepare_mono(&samples, MODEL_SAMPLE_RATE).unwrap();
        assert!(matches!(output, Cow::Borrowed(_)));
        assert_eq!(
            samples.map(f32::to_bits).as_slice(),
            output.iter().copied().map(f32::to_bits).collect::<Vec<_>>()
        );
    }

    #[test]
    fn invalid_pcm_is_rejected_even_at_native_rate() {
        for rate in [0, 22_050, 44_100] {
            for samples in [&[][..], &[f32::NAN], &[f32::INFINITY]] {
                assert!(prepare_mono(samples, rate).is_err());
            }
        }
        assert!(prepare_mono(&[1.0], 0).is_err());
    }

    #[test]
    fn impulse_alignment_and_tail_survive_multiple_rates() {
        for rate in [8_000, 16_000, 44_100, 48_000, 96_000, 192_000] {
            let len = rate as usize / 5;
            for position in [0, len / 2, len - 20] {
                let mut input = vec![0.0; len];
                input[position] = 1.0;
                let output = prepare_mono(&input, rate).unwrap();
                let expected =
                    (position * MODEL_SAMPLE_RATE as usize + rate as usize / 2) / rate as usize;
                assert_eq!(output.len(), 4410);
                assert!(
                    peak(&output).abs_diff(expected) <= 1,
                    "rate={rate} position={position} expected={expected} actual={}",
                    peak(&output)
                );
                assert!(
                    output[peak(&output)].abs() > 0.05,
                    "lost impulse at rate={rate} position={position}"
                );
                if rate == 44_100 {
                    assert_eq!(peak(&output), expected);
                }
            }
        }
    }

    #[test]
    fn short_and_chunk_boundary_lengths_are_exact_and_finite() {
        for rate in [8_000, 16_000, 44_100, 48_000, 96_000] {
            for len in [1, 2, 17, 127, 255, 256, 257, 4095, 4096, 4097, 8192, 8201] {
                let wanted = (len * MODEL_SAMPLE_RATE as usize + rate as usize / 2) / rate as usize;
                let input = vec![0.5; len];
                let result = prepare_mono(&input, rate);
                if wanted == 0 {
                    assert!(result.is_err());
                    continue;
                }
                let output = result.unwrap();
                assert_eq!(output.len(), wanted, "rate={rate} len={len}");
                assert!(output.iter().all(|value| value.is_finite()));
                if len > 127 {
                    assert!(output[output.len() / 2] > 0.4, "short signal was discarded");
                }
            }
        }
    }

    #[test]
    fn chunk_size_does_not_change_alignment_or_samples() {
        let input: Vec<f32> = (0..12345)
            .map(|index| if index % 701 == 0 { 1.0 } else { 0.0 })
            .collect();
        for rate in [16_000, 44_100, 48_000, 96_000] {
            let expected = resample(&input, rate, CHUNK_FRAMES).unwrap();
            for chunk in [256, 1024, 16384] {
                let actual = resample(&input, rate, chunk).unwrap();
                assert_eq!(actual.len(), expected.len());
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| (a - b).abs() < 1e-4),
                    "rate={rate} chunk={chunk}"
                );
            }
        }
    }

    #[test]
    fn downmix_validates_frames_and_avoids_overflow() {
        let mut output = Vec::new();
        append_mono(&mut output, &[0.25, 0.75, -0.5, 0.5], 2).unwrap();
        assert_eq!(output, [0.5, 0.0]);
        append_mono(&mut output, &[f32::MAX, f32::MAX], 2).unwrap();
        assert_eq!(output[2].to_bits(), f32::MAX.to_bits());
        assert!(append_mono(&mut output, &[1.0], 2).is_err());
        assert!(append_mono(&mut output, &[1.0], 0).is_err());
        assert!(append_mono(&mut output, &[f32::NAN], 1).is_err());
    }

    #[test]
    fn file_and_pcm_paths_share_preprocessing() {
        // Generated float WAV: no external audio or model fixture is needed.
        for rate in [22_050_u32, 44_100, 48_000, 96_000] {
            let mono: Vec<f32> = (0..8201)
                .map(|i| if i % 997 == 0 { 0.5 } else { 0.0 })
                .collect();
            let data_len = u32::try_from(mono.len() * 8).unwrap();
            let mut wav = Vec::new();
            wav.extend_from_slice(b"RIFF");
            wav.extend_from_slice(&(36 + data_len).to_le_bytes());
            wav.extend_from_slice(b"WAVEfmt ");
            wav.extend_from_slice(&16_u32.to_le_bytes());
            wav.extend_from_slice(&3_u16.to_le_bytes()); // IEEE float
            wav.extend_from_slice(&2_u16.to_le_bytes()); // stereo
            wav.extend_from_slice(&rate.to_le_bytes());
            wav.extend_from_slice(&(rate * 8).to_le_bytes());
            wav.extend_from_slice(&8_u16.to_le_bytes());
            wav.extend_from_slice(&32_u16.to_le_bytes());
            wav.extend_from_slice(b"data");
            wav.extend_from_slice(&data_len.to_le_bytes());
            for sample in &mono {
                wav.extend_from_slice(&(sample * 0.5).to_le_bytes());
                wav.extend_from_slice(&(sample * 1.5).to_le_bytes());
            }
            let path = std::env::temp_dir().join(format!(
                "rhythm-map-resample-{}-{rate}.wav",
                std::process::id()
            ));
            let mut file = File::create_new(&path).unwrap();
            std::io::Write::write_all(&mut file, &wav).unwrap();
            drop(file);
            let native = decode_native(&path).unwrap();
            assert_eq!(native.sample_rate, rate);
            assert_eq!(native.samples, mono);
            let result = decode(&path);
            std::fs::remove_file(&path).unwrap();
            let audio = result.unwrap();
            assert_eq!(audio.sample_rate, MODEL_SAMPLE_RATE);
            assert_eq!(audio.samples, prepare_mono(&mono, rate).unwrap().as_ref());
        }
    }
}

//! Adapter from the MIT-licensed Beat This Rust port to backend-neutral events.

use std::path::Path;

use beat_this::{BeatThis, RtenRuntime, Runtime};
use rhythm_map_core::{
    BackendError, ModelInfo, ObservedBeat, RhythmObservationBackend, RhythmObservations,
};

type DefaultModel = <RtenRuntime as Runtime>::Model;
const FRAME_RATE_HZ: f64 = 50.0;

/// Configurable peak-picking policy applied to Beat This frame logits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeakPickingOptions {
    /// Strict lower logit bound for a peak. Zero equals probability 0.5.
    pub logit_threshold: f32,
    /// Number of frames inspected on either side of a candidate maximum.
    pub local_max_radius_frames: usize,
    /// Adjacent peak indices at or below this distance are averaged together.
    pub deduplicate_width_frames: usize,
}

impl Default for PeakPickingOptions {
    fn default() -> Self {
        Self {
            logit_threshold: 0.0,
            local_max_radius_frames: 3,
            deduplicate_width_frames: 1,
        }
    }
}

/// Conservative sequence decoder for weak peaks between strong Beat This events.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SupportedMidpointOptions {
    /// Strict lower logit bound for weak candidate peaks.
    pub candidate_logit_threshold: f32,
    /// Maximum distance from an interval midpoint as a fraction of that interval.
    pub maximum_midpoint_offset_ratio: f64,
    /// Number of strong-beat gaps inspected on either side for run support.
    pub support_radius_gaps: usize,
    /// Minimum supported gaps required inside the local support window.
    pub minimum_supported_gaps: usize,
}

/// Viterbi beat-path options over Beat This frame logits.
///
/// The path models beat period and phase, but emitted events must still snap to
/// real local maxima in the model output. The decoder never emits a bare grid
/// timestamp.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SequencePathOptions {
    /// Strict lower logit bound for a local maximum to become an event.
    pub candidate_logit_threshold: f32,
    /// Local-maximum radius used for event candidates.
    pub candidate_local_max_radius_frames: usize,
    /// Maximum frame distance between a Viterbi beat state and a model peak.
    pub maximum_peak_correction_frames: usize,
    /// Slowest period represented by the path.
    pub minimum_bpm: f64,
    /// Fastest period represented by the path.
    pub maximum_bpm: f64,
    /// Squared log-period penalty applied when tempo changes at a beat.
    pub tempo_change_penalty: f64,
    /// Log-score prior added when the path enters a beat state.
    pub beat_state_bias: f64,
    /// Maximum path-beat gap joining events into one weak-event sequence.
    pub support_radius_beats: usize,
    /// Minimum weak candidates required in one connected sequence.
    pub minimum_supported_candidates: usize,
    /// Minimum weak candidates required in the local support radius around an
    /// emitted event.
    pub minimum_local_supported_candidates: usize,
    /// Require every recovered weak run to connect to the first or last
    /// model-supported beat-path event.
    pub require_edge_connection: bool,
    /// Maximum path-beat distance from a recovered run to an observed edge.
    pub maximum_edge_gap_beats: usize,
}

/// Deployable decoding policy used by [`BeatThisBackend`].
///
/// The upstream policy remains the default. Alternative policies are explicit
/// so calibration can exercise the complete product path without changing the
/// behavior of CLI, FFI, or WASM consumers.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum BeatThisDecoderPolicy {
    /// Match the decoder shipped by the upstream Rust port.
    #[default]
    Upstream,
    /// Apply an explicit peak-picking configuration.
    PeakPicking(PeakPickingOptions),
    /// Recover repeated weak model peaks between strong events.
    SupportedMidpoints(SupportedMidpointOptions),
    /// Decode a variable-tempo beat path and retain only supported model peaks.
    SequencePath(SequencePathOptions),
}

impl Default for SupportedMidpointOptions {
    fn default() -> Self {
        Self {
            candidate_logit_threshold: -3.0,
            maximum_midpoint_offset_ratio: 0.15,
            support_radius_gaps: 2,
            minimum_supported_gaps: 3,
        }
    }
}

impl Default for SequencePathOptions {
    fn default() -> Self {
        Self {
            candidate_logit_threshold: -3.0,
            candidate_local_max_radius_frames: 1,
            maximum_peak_correction_frames: 3,
            minimum_bpm: 40.0,
            maximum_bpm: 320.0,
            tempo_change_penalty: 100.0,
            beat_state_bias: 2.0,
            support_radius_beats: 3,
            minimum_supported_candidates: 6,
            minimum_local_supported_candidates: 3,
            require_edge_connection: true,
            maximum_edge_gap_beats: 2,
        }
    }
}

/// One model inference retained before discrete peak decoding.
#[derive(Debug, Clone)]
pub struct BeatThisInference {
    duration_s: f64,
    beat_logits: Vec<f32>,
    downbeat_logits: Vec<f32>,
}

impl BeatThisInference {
    /// Per-frame beat logits at 50 frames per second.
    #[must_use]
    pub fn beat_logits(&self) -> &[f32] {
        &self.beat_logits
    }

    /// Per-frame downbeat logits at 50 frames per second.
    #[must_use]
    pub fn downbeat_logits(&self) -> &[f32] {
        &self.downbeat_logits
    }
}

/// Decoded mono audio returned by the convenience file adapter.
#[derive(Debug, Clone)]
pub struct DecodedAudio {
    /// Mono PCM samples.
    pub samples: Vec<f32>,
    /// PCM sample rate.
    pub sample_rate: u32,
}

/// Beat This implementation of the observation boundary.
pub struct BeatThisBackend {
    tracker: BeatThis<DefaultModel>,
    model_name: String,
    decoder_policy: BeatThisDecoderPolicy,
}

impl BeatThisBackend {
    /// Load the frontend and beat/downbeat ONNX graphs using pure-Rust `rten`.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when either model cannot be loaded.
    pub fn load(
        mel_model_path: impl AsRef<Path>,
        beat_model_path: impl AsRef<Path>,
    ) -> Result<Self, BackendError> {
        let beat_model_path = beat_model_path.as_ref();
        let tracker = BeatThis::new(&RtenRuntime, mel_model_path.as_ref(), beat_model_path)
            .map_err(|error| {
                BackendError::new(format!("failed to load Beat This models: {error}"))
            })?;
        let model_name = beat_model_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("beat_this")
            .to_string();
        Ok(Self {
            tracker,
            model_name,
            decoder_policy: BeatThisDecoderPolicy::default(),
        })
    }

    /// Select an explicit deployable decoder policy.
    #[must_use]
    pub const fn with_decoder_policy(mut self, policy: BeatThisDecoderPolicy) -> Self {
        self.decoder_policy = policy;
        self
    }

    /// Run the neural frontend and model while retaining undecoded frame logits.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when model inference fails.
    pub fn infer_mono(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<BeatThisInference, BackendError> {
        if sample_rate == 0 {
            return Err(BackendError::new("sample rate must be greater than zero"));
        }
        let result = self
            .tracker
            .analyze_audio(samples, sample_rate)
            .map_err(|error| BackendError::new(format!("Beat This inference failed: {error}")))?;
        Ok(BeatThisInference {
            duration_s: usize_to_f64(samples.len()) / f64::from(sample_rate),
            beat_logits: result.beat_logits,
            downbeat_logits: result.downbeat_logits,
        })
    }

    /// Decode one retained inference with an explicit peak-picking policy.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] for a non-finite threshold or mismatched logits.
    pub fn decode_inference(
        &self,
        inference: &BeatThisInference,
        options: PeakPickingOptions,
    ) -> Result<RhythmObservations, BackendError> {
        if !options.logit_threshold.is_finite() {
            return Err(BackendError::new("peak logit threshold must be finite"));
        }
        if inference.beat_logits.len() != inference.downbeat_logits.len() {
            return Err(BackendError::new(
                "Beat This beat and downbeat logits have different lengths",
            ));
        }
        let beat_frames = find_peaks(&inference.beat_logits, options);
        let downbeat_frames = find_peaks(&inference.downbeat_logits, options);
        Ok(self.observations_from_frames(inference, &beat_frames, &downbeat_frames))
    }

    /// Decode upstream peaks and recover locally supported weak midpoint peaks.
    ///
    /// This candidate never invents a grid timestamp: every added event must be
    /// a radius-three local maximum above `candidate_logit_threshold`, close to
    /// the midpoint of two upstream beats, and part of a locally repeated run.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] for invalid options or mismatched logits.
    pub fn decode_inference_with_supported_midpoints(
        &self,
        inference: &BeatThisInference,
        options: SupportedMidpointOptions,
    ) -> Result<RhythmObservations, BackendError> {
        validate_midpoint_options(options)?;
        if inference.beat_logits.len() != inference.downbeat_logits.len() {
            return Err(BackendError::new(
                "Beat This beat and downbeat logits have different lengths",
            ));
        }
        let upstream_options = PeakPickingOptions::default();
        let upstream_beats = find_peaks(&inference.beat_logits, upstream_options);
        let candidate_options = PeakPickingOptions {
            logit_threshold: options.candidate_logit_threshold,
            ..upstream_options
        };
        let candidates = find_peaks(&inference.beat_logits, candidate_options);
        let beat_frames = recover_supported_midpoints(&upstream_beats, &candidates, options);
        let downbeat_frames = find_peaks(&inference.downbeat_logits, upstream_options);
        Ok(self.observations_from_frames(inference, &beat_frames, &downbeat_frames))
    }

    /// Decode a variable-tempo Viterbi path and add only locally supported
    /// model peaks that the upstream decoder missed.
    ///
    /// Upstream peaks are preserved. Every added event is a local maximum above
    /// `candidate_logit_threshold`, lies near a beat state in the best path,
    /// and belongs to a repeated weak-event sequence. A path state without a
    /// nearby model peak produces no timestamp.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] for invalid options or mismatched logits.
    pub fn decode_inference_with_sequence_path(
        &self,
        inference: &BeatThisInference,
        options: SequencePathOptions,
    ) -> Result<RhythmObservations, BackendError> {
        validate_sequence_path_options(options)?;
        if inference.beat_logits.len() != inference.downbeat_logits.len() {
            return Err(BackendError::new(
                "Beat This beat and downbeat logits have different lengths",
            ));
        }
        let upstream_options = PeakPickingOptions::default();
        let upstream_beats = find_peaks(&inference.beat_logits, upstream_options);
        let beat_frames = decode_sequence_path(&inference.beat_logits, &upstream_beats, options);
        let downbeat_frames = find_peaks(&inference.downbeat_logits, upstream_options);
        Ok(self.observations_from_frames(inference, &beat_frames, &downbeat_frames))
    }

    /// Decode retained logits through one explicit deployable policy.
    ///
    /// This is the shared dispatch point for live backend observation and
    /// single-inference evaluation, preventing those paths from interpreting a
    /// registered policy differently.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the selected policy or inference is invalid.
    pub fn decode_inference_with_policy(
        &self,
        inference: &BeatThisInference,
        policy: BeatThisDecoderPolicy,
    ) -> Result<RhythmObservations, BackendError> {
        match policy {
            BeatThisDecoderPolicy::Upstream => {
                self.decode_inference(inference, PeakPickingOptions::default())
            }
            BeatThisDecoderPolicy::PeakPicking(options) => {
                self.decode_inference(inference, options)
            }
            BeatThisDecoderPolicy::SupportedMidpoints(options) => {
                self.decode_inference_with_supported_midpoints(inference, options)
            }
            BeatThisDecoderPolicy::SequencePath(options) => {
                self.decode_inference_with_sequence_path(inference, options)
            }
        }
    }

    fn observations_from_frames(
        &self,
        inference: &BeatThisInference,
        beat_frames: &[f64],
        downbeat_frames: &[f64],
    ) -> RhythmObservations {
        let beats = beat_frames
            .iter()
            .map(|&frame| frame_to_time(frame))
            .collect::<Vec<_>>();
        let mut downbeats = downbeat_frames
            .iter()
            .map(|&frame| frame_to_time(frame))
            .collect::<Vec<_>>();
        snap_to_beats(&beats, &mut downbeats);

        RhythmObservations {
            duration_s: inference.duration_s,
            beats: observations_from_events(
                &beats,
                &downbeats,
                &inference.beat_logits,
                &inference.downbeat_logits,
            ),
            activity: Vec::new(),
            source: ModelInfo {
                backend: "beat-this-rten".to_string(),
                model: self.model_name.clone(),
                version: None,
                frame_rate_hz: Some(FRAME_RATE_HZ),
            },
        }
    }
}

impl RhythmObservationBackend for BeatThisBackend {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn observe_mono(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<RhythmObservations, BackendError> {
        let inference = self.infer_mono(samples, sample_rate)?;
        self.decode_inference_with_policy(&inference, self.decoder_policy)
    }
}

/// Decode a supported audio file to mono PCM for CLI and GUI adapters.
///
/// # Errors
///
/// Returns [`BackendError`] when the file cannot be decoded or resampled.
pub fn decode_audio(path: impl AsRef<Path>) -> Result<DecodedAudio, BackendError> {
    let audio = beat_this::load_audio(path.as_ref(), 22_050)
        .map_err(|error| BackendError::new(format!("failed to decode audio: {error}")))?;
    Ok(DecodedAudio {
        samples: audio.samples,
        sample_rate: audio.sample_rate,
    })
}

fn sigmoid(value: f32) -> f64 {
    1.0 / (1.0 + (-f64::from(value)).exp())
}

#[allow(clippy::cast_possible_truncation)]
fn frame_to_time(frame: f64) -> f64 {
    f64::from((frame / FRAME_RATE_HZ) as f32)
}

fn find_peaks(logits: &[f32], options: PeakPickingOptions) -> Vec<f64> {
    let candidates = logits
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| {
            if value <= options.logit_threshold {
                return None;
            }
            let start = index.saturating_sub(options.local_max_radius_frames);
            let end = index
                .saturating_add(options.local_max_radius_frames)
                .saturating_add(1)
                .min(logits.len());
            logits[start..end]
                .iter()
                .all(|&neighbor| neighbor <= value)
                .then_some(index)
        })
        .collect::<Vec<_>>();
    deduplicate_peaks(&candidates, options.deduplicate_width_frames)
}

fn validate_midpoint_options(options: SupportedMidpointOptions) -> Result<(), BackendError> {
    if !options.candidate_logit_threshold.is_finite() {
        return Err(BackendError::new(
            "midpoint candidate logit threshold must be finite",
        ));
    }
    if !options.maximum_midpoint_offset_ratio.is_finite()
        || !(0.0..=0.5).contains(&options.maximum_midpoint_offset_ratio)
    {
        return Err(BackendError::new(
            "maximum midpoint offset ratio must be finite and between zero and 0.5",
        ));
    }
    if options.minimum_supported_gaps == 0 {
        return Err(BackendError::new(
            "minimum supported midpoint gaps must be greater than zero",
        ));
    }
    Ok(())
}

fn validate_sequence_path_options(options: SequencePathOptions) -> Result<(), BackendError> {
    if !options.candidate_logit_threshold.is_finite() {
        return Err(BackendError::new(
            "sequence candidate logit threshold must be finite",
        ));
    }
    if !options.minimum_bpm.is_finite()
        || !options.maximum_bpm.is_finite()
        || options.minimum_bpm <= 0.0
        || options.maximum_bpm < options.minimum_bpm
    {
        return Err(BackendError::new(
            "sequence BPM range must be finite, positive, and ordered",
        ));
    }
    if !options.tempo_change_penalty.is_finite() || options.tempo_change_penalty < 0.0 {
        return Err(BackendError::new(
            "sequence tempo change penalty must be finite and non-negative",
        ));
    }
    if !options.beat_state_bias.is_finite() {
        return Err(BackendError::new("sequence beat-state bias must be finite"));
    }
    if options.minimum_supported_candidates == 0 || options.minimum_local_supported_candidates == 0
    {
        return Err(BackendError::new(
            "minimum sequence support counts must be greater than zero",
        ));
    }
    let (minimum_period, maximum_period) = period_frame_bounds(options);
    if minimum_period == 0
        || minimum_period > maximum_period
        || maximum_period > usize::from(u16::MAX)
    {
        return Err(BackendError::new(
            "sequence BPM range does not contain a representable frame period",
        ));
    }
    Ok(())
}

fn decode_sequence_path(
    logits: &[f32],
    upstream_beats: &[f64],
    options: SequencePathOptions,
) -> Vec<f64> {
    if logits.is_empty() {
        return upstream_beats.to_vec();
    }
    let candidate_options = PeakPickingOptions {
        logit_threshold: options.candidate_logit_threshold,
        local_max_radius_frames: options.candidate_local_max_radius_frames,
        deduplicate_width_frames: 1,
    };
    let candidates = find_peaks(logits, candidate_options);
    let path = viterbi_beat_path(logits, options);
    let snapped =
        snap_path_to_candidates(&path, &candidates, options.maximum_peak_correction_frames);
    let additions = supported_path_additions(
        &snapped,
        upstream_beats,
        options.support_radius_beats,
        options.minimum_supported_candidates,
        options.minimum_local_supported_candidates,
        options.require_edge_connection,
        options.maximum_edge_gap_beats,
    );
    let mut beats = upstream_beats.to_vec();
    beats.extend(additions);
    beats.sort_by(f64::total_cmp);
    beats.dedup_by(|left, right| (*left - *right).abs() <= 1.0);
    beats
}

fn bpm_to_period_frames(bpm: f64) -> f64 {
    60.0 * FRAME_RATE_HZ / bpm
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn period_frame_bounds(options: SequencePathOptions) -> (usize, usize) {
    (
        bpm_to_period_frames(options.maximum_bpm).ceil() as usize,
        bpm_to_period_frames(options.minimum_bpm).floor() as usize,
    )
}

fn viterbi_beat_path(logits: &[f32], options: SequencePathOptions) -> Vec<usize> {
    let (minimum_period, maximum_period) = period_frame_bounds(options);
    let periods = (minimum_period..=maximum_period).collect::<Vec<_>>();
    let mut offsets = Vec::with_capacity(periods.len());
    let mut total_states = 0;
    for &period in &periods {
        offsets.push(total_states);
        total_states += period;
    }

    let emission_at_beat = log_sigmoid(logits[0]) + options.beat_state_bias;
    let emission_off_beat = log_sigmoid(-logits[0]);
    let mut scores = vec![f64::NEG_INFINITY; total_states];
    for (&period, &offset) in periods.iter().zip(&offsets) {
        scores[offset] = emission_at_beat;
        scores[offset + 1..offset + period].fill(emission_off_beat);
    }
    let mut back_periods = vec![u16::MAX; logits.len() * periods.len()];

    for (frame, &logit) in logits.iter().enumerate().skip(1) {
        let emission_at_beat = log_sigmoid(logit) + options.beat_state_bias;
        let emission_off_beat = log_sigmoid(-logit);
        let mut next = vec![f64::NEG_INFINITY; total_states];
        for (period_index, (&period, &offset)) in periods.iter().zip(&offsets).enumerate() {
            for phase in 1..period {
                next[offset + phase] = scores[offset + phase - 1] + emission_off_beat;
            }
            let mut best_transition = f64::NEG_INFINITY;
            let mut best_period_index = 0;
            for (source_index, (&source_period, &source_offset)) in
                periods.iter().zip(&offsets).enumerate()
            {
                let log_ratio = (usize_to_f64(period) / usize_to_f64(source_period)).ln();
                let transition_penalty = options.tempo_change_penalty * log_ratio * log_ratio;
                let score = scores[source_offset + source_period - 1] - transition_penalty;
                if score > best_transition {
                    best_transition = score;
                    best_period_index = source_index;
                }
            }
            next[offset] = best_transition + emission_at_beat;
            back_periods[frame * periods.len() + period_index] =
                u16::try_from(best_period_index).expect("period state count fits u16");
        }
        scores = next;
    }

    let mut period_index = 0;
    let mut phase = 0;
    let mut terminal_best = f64::NEG_INFINITY;
    for (candidate_period, (&period, &offset)) in periods.iter().zip(&offsets).enumerate() {
        for candidate_phase in 0..period {
            let score = scores[offset + candidate_phase];
            if score > terminal_best {
                period_index = candidate_period;
                phase = candidate_phase;
                terminal_best = score;
            }
        }
    }
    let mut beats = Vec::new();
    for frame in (0..logits.len()).rev() {
        if phase == 0 {
            beats.push(frame);
        }
        if frame == 0 {
            break;
        }
        if phase == 0 {
            period_index = usize::from(back_periods[frame * periods.len() + period_index]);
            phase = periods[period_index] - 1;
        } else {
            phase -= 1;
        }
    }
    beats.reverse();
    beats
}

fn log_sigmoid(value: f32) -> f64 {
    let value = f64::from(value);
    if value >= 0.0 {
        -(-value).exp().ln_1p()
    } else {
        value - value.exp().ln_1p()
    }
}

fn snap_path_to_candidates(
    path: &[usize],
    candidates: &[f64],
    maximum_offset_frames: usize,
) -> Vec<(usize, f64)> {
    let maximum_offset = usize_to_f64(maximum_offset_frames);
    let mut last = f64::NEG_INFINITY;
    path.iter()
        .enumerate()
        .filter_map(|(path_index, &frame)| {
            let frame = usize_to_f64(frame);
            let candidate = candidates
                .iter()
                .copied()
                .filter(|&candidate| candidate > last)
                .filter(|&candidate| (candidate - frame).abs() <= maximum_offset)
                .min_by(|left, right| (*left - frame).abs().total_cmp(&(*right - frame).abs()))?;
            last = candidate;
            Some((path_index, candidate))
        })
        .collect()
}

fn supported_path_additions(
    snapped: &[(usize, f64)],
    upstream_beats: &[f64],
    support_radius_beats: usize,
    minimum_supported_candidates: usize,
    minimum_local_supported_candidates: usize,
    require_edge_connection: bool,
    maximum_edge_gap_beats: usize,
) -> Vec<f64> {
    let additions = snapped
        .iter()
        .map(|&(path_index, candidate)| {
            let is_upstream = upstream_beats
                .iter()
                .any(|&upstream| (upstream - candidate).abs() <= 1.0);
            (path_index, candidate, !is_upstream)
        })
        .collect::<Vec<_>>();
    let addition_indices = additions
        .iter()
        .filter_map(|&(path_index, _, is_addition)| is_addition.then_some(path_index))
        .collect::<Vec<_>>();
    let edge_connected_range = require_edge_connection
        .then(|| {
            edge_connected_addition_range(&additions, support_radius_beats, maximum_edge_gap_beats)
        })
        .flatten();
    additions
        .iter()
        .filter_map(|&(path_index, candidate, is_addition)| {
            if !is_addition {
                return None;
            }
            if require_edge_connection
                && !edge_connected_range.is_some_and(|(prefix_end, suffix_start)| {
                    path_index <= prefix_end || path_index >= suffix_start
                })
            {
                return None;
            }
            let supported =
                connected_addition_count(&addition_indices, path_index, support_radius_beats);
            let locally_supported = addition_indices
                .iter()
                .filter(|&&neighbor_index| {
                    neighbor_index.abs_diff(path_index) <= support_radius_beats
                })
                .count();
            (supported >= minimum_supported_candidates
                && locally_supported >= minimum_local_supported_candidates)
                .then_some(candidate)
        })
        .collect()
}

fn connected_addition_count(
    addition_indices: &[usize],
    target: usize,
    maximum_gap_beats: usize,
) -> usize {
    let Some(mut first) = addition_indices.iter().position(|&index| index == target) else {
        return 0;
    };
    let mut last = first;
    while first > 0
        && addition_indices[first].abs_diff(addition_indices[first - 1]) <= maximum_gap_beats
    {
        first -= 1;
    }
    while last + 1 < addition_indices.len()
        && addition_indices[last + 1].abs_diff(addition_indices[last]) <= maximum_gap_beats
    {
        last += 1;
    }
    last - first + 1
}

fn edge_connected_addition_range(
    additions: &[(usize, f64, bool)],
    support_radius_beats: usize,
    maximum_edge_gap_beats: usize,
) -> Option<(usize, usize)> {
    let first_snapped = additions.first()?.0;
    let last_snapped = additions.last()?.0;
    let addition_indices = additions
        .iter()
        .filter_map(|&(path_index, _, is_addition)| is_addition.then_some(path_index))
        .collect::<Vec<_>>();
    let first_addition = *addition_indices.first()?;
    let last_addition = *addition_indices.last()?;

    let mut prefix_end = usize::MIN;
    if first_addition.abs_diff(first_snapped) <= maximum_edge_gap_beats {
        prefix_end = first_addition;
        for pair in addition_indices.windows(2) {
            if pair[1].abs_diff(pair[0]) > support_radius_beats {
                break;
            }
            prefix_end = pair[1];
        }
    }

    let mut suffix_start = usize::MAX;
    if last_addition.abs_diff(last_snapped) <= maximum_edge_gap_beats {
        suffix_start = last_addition;
        for pair in addition_indices.windows(2).rev() {
            if pair[1].abs_diff(pair[0]) > support_radius_beats {
                break;
            }
            suffix_start = pair[0];
        }
    }

    Some((prefix_end, suffix_start))
}

fn recover_supported_midpoints(
    upstream_beats: &[f64],
    candidates: &[f64],
    options: SupportedMidpointOptions,
) -> Vec<f64> {
    let supported = upstream_beats
        .windows(2)
        .map(|pair| midpoint_candidate(pair[0], pair[1], candidates, options))
        .collect::<Vec<_>>();
    let mut recovered = upstream_beats.to_vec();
    for (gap_index, candidate) in supported.iter().enumerate() {
        let Some(candidate) = candidate else {
            continue;
        };
        let start = gap_index.saturating_sub(options.support_radius_gaps);
        let end = gap_index
            .saturating_add(options.support_radius_gaps)
            .saturating_add(1)
            .min(supported.len());
        let support_count = supported[start..end]
            .iter()
            .filter(|candidate| candidate.is_some())
            .count();
        if support_count >= options.minimum_supported_gaps {
            recovered.push(*candidate);
        }
    }
    recovered.sort_by(f64::total_cmp);
    recovered.dedup();
    recovered
}

fn midpoint_candidate(
    left: f64,
    right: f64,
    candidates: &[f64],
    options: SupportedMidpointOptions,
) -> Option<f64> {
    let gap = right - left;
    if gap <= 2.0 {
        return None;
    }
    let midpoint = left.midpoint(right);
    let maximum_offset = gap * options.maximum_midpoint_offset_ratio;
    candidates
        .iter()
        .copied()
        .filter(|&candidate| candidate > left + 1.0 && candidate < right - 1.0)
        .filter(|&candidate| (candidate - midpoint).abs() <= maximum_offset)
        .min_by(|left, right| {
            (*left - midpoint)
                .abs()
                .total_cmp(&(*right - midpoint).abs())
        })
}

fn deduplicate_peaks(peaks: &[usize], width: usize) -> Vec<f64> {
    let Some(&first) = peaks.first() else {
        return Vec::new();
    };
    let mut result = Vec::new();
    let mut running_mean = usize_to_f64(first);
    let mut count = 1.0;
    for &candidate in &peaks[1..] {
        let candidate = usize_to_f64(candidate);
        if candidate - running_mean <= usize_to_f64(width) {
            count += 1.0;
            running_mean += (candidate - running_mean) / count;
        } else {
            result.push(running_mean);
            running_mean = candidate;
            count = 1.0;
        }
    }
    result.push(running_mean);
    result
}

fn snap_to_beats(beats: &[f64], downbeats: &mut Vec<f64>) {
    for downbeat in downbeats.iter_mut() {
        if let Some(beat) = beats.iter().min_by(|left, right| {
            (*left - *downbeat)
                .abs()
                .total_cmp(&(*right - *downbeat).abs())
        }) {
            *downbeat = *beat;
        }
    }
    downbeats.sort_by(f64::total_cmp);
    downbeats.dedup();
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn observations_from_events(
    beats: &[f64],
    downbeats: &[f64],
    beat_logits: &[f32],
    downbeat_logits: &[f32],
) -> Vec<ObservedBeat> {
    beats
        .iter()
        .map(|&time_s| {
            let frame = ((time_s * FRAME_RATE_HZ).round() as usize)
                .min(beat_logits.len().saturating_sub(1));
            let downbeat = downbeats
                .iter()
                .any(|&candidate| (candidate - time_s).abs() <= 0.07);
            ObservedBeat {
                time_s,
                confidence: beat_logits.get(frame).copied().map_or(0.5, sigmoid),
                downbeat_confidence: if downbeat {
                    downbeat_logits
                        .get(frame)
                        .copied()
                        .map_or(0.75, sigmoid)
                        .max(0.5)
                } else {
                    0.0
                },
            }
        })
        .collect()
}

#[allow(clippy::cast_precision_loss)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_peak_picker_matches_upstream_rules() {
        let mut logits = vec![-1.0; 40];
        logits[5] = 2.0;
        logits[20] = 1.0;
        logits[21] = 1.0;
        logits[30] = -0.1;

        assert_eq!(
            find_peaks(&logits, PeakPickingOptions::default()),
            vec![5.0, 20.5]
        );
    }

    #[test]
    fn lower_threshold_recovers_subzero_local_peak() {
        let logits = [-2.0, -0.5, -2.0];
        let options = PeakPickingOptions {
            logit_threshold: -1.0,
            ..PeakPickingOptions::default()
        };

        assert_eq!(find_peaks(&logits, options), vec![1.0]);
    }

    #[test]
    fn narrower_local_max_window_retains_nearby_peaks() {
        let logits = [-2.0, 1.0, -1.0, -1.0, 0.5, -2.0];
        let narrow = PeakPickingOptions {
            local_max_radius_frames: 1,
            ..PeakPickingOptions::default()
        };

        assert_eq!(find_peaks(&logits, narrow), vec![1.0, 4.0]);
        assert_eq!(
            find_peaks(&logits, PeakPickingOptions::default()),
            vec![1.0]
        );
    }

    #[test]
    fn extreme_radius_does_not_overflow() {
        let logits = [-1.0, 1.0, -1.0];
        let options = PeakPickingOptions {
            local_max_radius_frames: usize::MAX,
            ..PeakPickingOptions::default()
        };

        assert_eq!(find_peaks(&logits, options), vec![1.0]);
    }

    #[test]
    fn supported_midpoint_run_recovers_real_candidate_peaks() {
        let upstream = [0.0, 20.0, 40.0, 60.0, 80.0];
        let candidates = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

        assert_eq!(
            recover_supported_midpoints(
                &upstream,
                &candidates,
                SupportedMidpointOptions::default()
            ),
            candidates
        );
    }

    #[test]
    fn isolated_midpoint_candidate_is_not_recovered() {
        let upstream = [0.0, 20.0, 40.0, 60.0, 80.0];
        let candidates = [0.0, 20.0, 30.0, 40.0, 60.0, 80.0];

        assert_eq!(
            recover_supported_midpoints(
                &upstream,
                &candidates,
                SupportedMidpointOptions::default()
            ),
            upstream
        );
    }

    #[test]
    fn sequence_path_recovers_repeated_weak_peaks_through_track_end() {
        let mut logits = vec![-10.0; 260];
        for frame in (10..=235).step_by(25) {
            logits[frame] = if frame <= 85 { 4.0 } else { -1.5 };
        }
        let upstream = find_peaks(&logits, PeakPickingOptions::default());
        let decoded = decode_sequence_path(&logits, &upstream, SequencePathOptions::default());

        assert_eq!(upstream, vec![10.0, 35.0, 60.0, 85.0]);
        assert_eq!(
            decoded,
            (10..=235).step_by(25).map(usize_to_f64).collect::<Vec<_>>()
        );
    }

    #[test]
    fn sequence_path_never_emits_a_grid_time_without_a_model_peak() {
        let mut logits = vec![-10.0; 260];
        for frame in [10, 35, 60, 85, 210, 235] {
            logits[frame] = 4.0;
        }
        let upstream = find_peaks(&logits, PeakPickingOptions::default());

        let decoded = decode_sequence_path(&logits, &upstream, SequencePathOptions::default());

        assert_eq!(decoded, upstream);
        assert!(!decoded.contains(&110.0));
    }

    #[test]
    fn sequence_path_rejects_an_interior_weak_peak_run() {
        let mut logits = vec![-10.0; 360];
        for frame in (10..=335).step_by(25) {
            logits[frame] = if (110..=210).contains(&frame) {
                -1.5
            } else {
                4.0
            };
        }
        let upstream = find_peaks(&logits, PeakPickingOptions::default());

        let decoded = decode_sequence_path(&logits, &upstream, SequencePathOptions::default());

        assert_eq!(decoded, upstream);
        assert!(!decoded.contains(&160.0));
    }

    #[test]
    fn sequence_path_follows_a_supported_real_tempo_doubling() {
        let mut logits = vec![-10.0; 360];
        let expected = [
            10, 40, 70, 100, 130, 145, 160, 175, 190, 205, 220, 235, 250, 265, 280,
        ];
        for frame in expected {
            logits[frame] = 4.0;
        }
        let upstream = find_peaks(&logits, PeakPickingOptions::default());

        let decoded = decode_sequence_path(&logits, &upstream, SequencePathOptions::default());

        assert_eq!(decoded, expected.map(usize_to_f64));
    }

    #[test]
    fn sequence_path_rejects_invalid_bpm_ranges() {
        let error = validate_sequence_path_options(SequencePathOptions {
            minimum_bpm: 200.0,
            maximum_bpm: 100.0,
            ..SequencePathOptions::default()
        })
        .unwrap_err();

        assert!(error.to_string().contains("BPM range"));
    }
}

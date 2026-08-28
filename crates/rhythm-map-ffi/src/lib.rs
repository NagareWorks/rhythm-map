//! C ABI that exposes opaque analyzers and owned JSON results.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::ptr;
use std::slice;
use std::sync::Mutex;

use rhythm_map_beat_this::BeatThisBackend;
use rhythm_map_core::Engine;
use rhythm_map_models::{ModelArtifactRole, verify_model_pack};

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

/// Opaque analyzer owned by a C caller.
pub struct RhythmMapAnalyzer {
    engine: Mutex<Engine<BeatThisBackend>>,
}

/// Return the ABI version.
#[unsafe(no_mangle)]
pub const extern "C" fn rhythm_map_abi_version() -> u32 {
    1
}

/// Create an analyzer from null-terminated UTF-8 model paths.
///
/// # Safety
///
/// Both pointers must be non-null and point to valid null-terminated UTF-8
/// strings for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rhythm_map_analyzer_new(
    mel_model_path: *const c_char,
    beat_model_path: *const c_char,
) -> *mut RhythmMapAnalyzer {
    ffi_ptr(|| {
        let mel = c_path(mel_model_path, "mel model path")?;
        let beat = c_path(beat_model_path, "beat model path")?;
        let backend = BeatThisBackend::load(&mel, &beat).map_err(|error| error.to_string())?;
        Ok(new_analyzer(backend))
    })
}

/// Create an analyzer from a verified model-pack manifest and artifact root.
///
/// This is the preferred constructor for new integrations. Every artifact is
/// checked against the manifest before model loading, and the pack identity is
/// retained in the returned analysis metadata.
///
/// # Safety
///
/// Both pointers must be non-null and point to valid null-terminated UTF-8
/// strings for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rhythm_map_analyzer_new_from_model_pack(
    manifest_path: *const c_char,
    artifact_root: *const c_char,
) -> *mut RhythmMapAnalyzer {
    ffi_ptr(|| {
        let manifest_path = c_path(manifest_path, "model-pack manifest path")?;
        let artifact_root = c_path(artifact_root, "model artifact root")?;
        let pack =
            verify_model_pack(&manifest_path, &artifact_root).map_err(|error| error.to_string())?;
        let mel_model = pack
            .path_for(ModelArtifactRole::MelFrontend)
            .ok_or_else(|| "verified model pack has no mel_frontend artifact".to_string())?;
        let beat_model = pack
            .path_for(ModelArtifactRole::BeatModel)
            .ok_or_else(|| "verified model pack has no beat_model artifact".to_string())?;
        let model_name = pack.manifest().id.clone();
        let model_version = Some(format!("manifest-sha256:{}", pack.manifest_sha256()));
        let backend = BeatThisBackend::load_with_model_identity(
            mel_model,
            beat_model,
            model_name,
            model_version,
        )
        .map_err(|error| error.to_string())?;
        Ok(new_analyzer(backend))
    })
}

/// Analyze interleaved `f32` PCM and return owned UTF-8 JSON.
///
/// Free the returned pointer with [`rhythm_map_string_free`].
///
/// # Safety
///
/// `analyzer` must be a live pointer returned by
/// [`rhythm_map_analyzer_new`]. When `sample_count` is non-zero, `samples` must
/// reference at least that many initialized `f32` values for the duration of
/// this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rhythm_map_analyze_pcm_json(
    analyzer: *mut RhythmMapAnalyzer,
    samples: *const f32,
    sample_count: usize,
    sample_rate: u32,
    channels: u16,
) -> *mut c_char {
    ffi_ptr(|| {
        let analyzer =
            unsafe { analyzer.as_ref() }.ok_or_else(|| "analyzer pointer is null".to_string())?;
        if samples.is_null() && sample_count != 0 {
            return Err("sample pointer is null".to_string());
        }
        let samples = if sample_count == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(samples, sample_count) }
        };
        let mut engine = analyzer
            .engine
            .lock()
            .map_err(|_| "analyzer lock is poisoned".to_string())?;
        let analysis = engine
            .analyze_pcm(samples, sample_rate, channels)
            .map_err(|error| error.to_string())?;
        let json = serde_json::to_string(&analysis).map_err(|error| error.to_string())?;
        CString::new(json)
            .map(CString::into_raw)
            .map_err(|error| error.to_string())
    })
}

/// Return the latest error on the current thread.
///
/// The pointer remains valid until another API call on the same thread.
#[unsafe(no_mangle)]
pub extern "C" fn rhythm_map_last_error() -> *const c_char {
    LAST_ERROR.with(|message| message.borrow().as_ptr())
}

/// Release a JSON string returned by this library.
///
/// # Safety
///
/// `value` must be null or a pointer returned by
/// [`rhythm_map_analyze_pcm_json`] that has not already been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rhythm_map_string_free(value: *mut c_char) {
    if !value.is_null() {
        drop(unsafe { CString::from_raw(value) });
    }
}

/// Release an analyzer.
///
/// # Safety
///
/// `analyzer` must be null or a live pointer returned by
/// [`rhythm_map_analyzer_new`] that has not already been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rhythm_map_analyzer_free(analyzer: *mut RhythmMapAnalyzer) {
    if !analyzer.is_null() {
        drop(unsafe { Box::from_raw(analyzer) });
    }
}

fn ffi_ptr<T>(operation: impl FnOnce() -> Result<*mut T, String>) -> *mut T {
    clear_error();
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(pointer)) => pointer,
        Ok(Err(error)) => {
            set_error(&error);
            ptr::null_mut()
        }
        Err(_) => {
            set_error("panic crossed the internal FFI boundary");
            ptr::null_mut()
        }
    }
}

fn new_analyzer(backend: BeatThisBackend) -> *mut RhythmMapAnalyzer {
    Box::into_raw(Box::new(RhythmMapAnalyzer {
        engine: Mutex::new(Engine::new(backend)),
    }))
}

fn c_path(value: *const c_char, name: &str) -> Result<PathBuf, String> {
    if value.is_null() {
        return Err(format!("{name} is null"));
    }
    let text = unsafe { CStr::from_ptr(value) }
        .to_str()
        .map_err(|error| format!("{name} is not UTF-8: {error}"))?;
    Ok(PathBuf::from(text))
}

fn clear_error() {
    LAST_ERROR.with(|message| *message.borrow_mut() = CString::default());
}

fn set_error(message: &str) {
    let sanitized = message.replace('\0', " ");
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(sanitized).unwrap_or_default();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_is_stable() {
        assert_eq!(rhythm_map_abi_version(), 1);
    }

    #[test]
    fn model_pack_constructor_reports_null_manifest() {
        let analyzer =
            unsafe { rhythm_map_analyzer_new_from_model_pack(ptr::null(), c"models".as_ptr()) };

        assert!(analyzer.is_null());
        let error = unsafe { CStr::from_ptr(rhythm_map_last_error()) };
        assert_eq!(error.to_str().unwrap(), "model-pack manifest path is null");
    }

    #[test]
    fn analyze_reports_null_analyzer() {
        let json =
            unsafe { rhythm_map_analyze_pcm_json(ptr::null_mut(), ptr::null(), 0, 44_100, 1) };

        assert!(json.is_null());
        let error = unsafe { CStr::from_ptr(rhythm_map_last_error()) };
        assert_eq!(error.to_str().unwrap(), "analyzer pointer is null");
    }
}

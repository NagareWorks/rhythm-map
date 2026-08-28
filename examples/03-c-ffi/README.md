# 03 - Call Rhythm Map through the C ABI

This example builds one native library and calls the same ABI from C, Python,
C#, and Unity. All callers receive the complete schema-versioned analysis JSON;
none of the bindings reimplement tempo estimation or expose musical strategy
switches.

New integrations should construct an analyzer with
`rhythm_map_analyzer_new_from_model_pack`. It verifies artifact sizes and
SHA-256 digests and preserves the model-pack ID and manifest digest in the
analysis source metadata. The older two-model-path constructor remains ABI
compatible, but deliberately is not used here.

## Build the native library

From the repository root:

```bash
cargo build -p rhythm-map-ffi --release
```

The dynamic library is written to the Cargo target directory:

| Platform | Library |
| --- | --- |
| Windows | `rhythm_map_ffi.dll` |
| Linux | `librhythm_map_ffi.so` |
| macOS | `librhythm_map_ffi.dylib` |

Prepare the model artifacts as described by
[`02-audio-file`](../02-audio-file/README.md). The C, Python, and standalone C#
WAV runners accept uncompressed PCM16 WAV input so that their decoders stay
small. The ABI itself accepts any caller-owned interleaved `float` PCM; Unity
therefore passes `AudioClip` data without a file decode step.

## Ownership and errors

| Value | Owner | Release rule |
| --- | --- | --- |
| input PCM | caller | remains valid through the analysis call |
| `RhythmMapAnalyzer *` | caller | call `rhythm_map_analyzer_free` once |
| returned JSON `char *` | caller | copy it, then call `rhythm_map_string_free` once |
| `rhythm_map_last_error()` | library/thread | borrowed; copy it and never free it |

Constructors and analysis return null on failure. Read
`rhythm_map_last_error()` immediately on the same thread. Rust panics are
caught inside the library and never unwind through foreign code. Analyzer
calls are serialized internally; separate analyzers can be owned by separate
workers.

`sample_count` is the number of interleaved float values, not the number of
frames. For example, one second of stereo 48 kHz PCM has 96,000 samples.

## C

[`c/main.c`](c/main.c) includes a small PCM16/float32 WAV reader and demonstrates
every ownership edge. Configure it against the directory containing the Cargo
library (or its import library on Windows):

```bash
cmake -S examples/03-c-ffi/c -B build/03-c-ffi \
  -DRHYTHM_MAP_LIBRARY_DIR="$PWD/target/release"
cmake --build build/03-c-ffi --config Release
```

On Windows, put the DLL beside the executable or add its directory to `PATH`.
On Linux or macOS, install it in a standard location, configure an rpath, or
add its directory to `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH`. Then run:

```bash
./build/03-c-ffi/rhythm_map_c_example \
  models/beat-this-full-v1.json /path/to/beat-this-full-v1 song.wav
```

Use `rhythm_map_c_example --abi-only` as a model-free linkage smoke test.

## Python (`ctypes`)

[`python/rhythm_map.py`](python/rhythm_map.py) is a dependency-free binding with
context-managed analyzer ownership. [`python/analyze_wav.py`](python/analyze_wav.py)
shows one call returning an ordinary Python dictionary:

```bash
python examples/03-c-ffi/python/analyze_wav.py \
  --library target/release/librhythm_map_ffi.so \
  models/beat-this-full-v1.json /path/to/beat-this-full-v1 song.wav
```

On Windows, pass `target\release\rhythm_map_ffi.dll`. A model-free loader check
is also available:

```bash
python examples/03-c-ffi/python/analyze_wav.py \
  --library target/release/librhythm_map_ffi.so --abi-only
```

## C# and Unity

[`csharp/RhythmMapNative.cs`](csharp/RhythmMapNative.cs) is the reusable P/Invoke
wrapper. It uses `SafeHandle` for the analyzer, copies UTF-8 JSON before freeing
the native allocation, and converts native null returns into
`RhythmMapException`.

The standalone runner can be built and called with:

```bash
dotnet build examples/03-c-ffi/csharp/RhythmMapExample.csproj -c Release
dotnet run --project examples/03-c-ffi/csharp/RhythmMapExample.csproj -c Release -- \
  models/beat-this-full-v1.json /path/to/beat-this-full-v1 song.wav
```

The native library must be discoverable by the process. Copy it beside the
generated executable or set the platform library search path. Run the compiled
program with `--abi-only` for a model-free P/Invoke smoke test.

For a Windows, Linux, or macOS desktop project using Unity 2021.3 or newer:

1. Copy `RhythmMapNative.cs` and
   [`UnityAudioClipExample.cs`](csharp/UnityAudioClipExample.cs) into
   `Assets/Scripts`.
2. Copy the platform library into `Assets/Plugins/x86_64` and configure its
   Unity platform/import settings. Use `rhythm_map_ffi.dll` on Windows,
   `librhythm_map_ffi.so` on Linux, or `librhythm_map_ffi.dylib` on macOS.
3. Put the checked-in manifest and downloaded artifacts below
   `Assets/StreamingAssets/models` using the paths in the example component.
4. Make the assigned `AudioClip` readable, normally with **Decompress On Load**,
   so `AudioClip.GetData` returns real interleaved PCM.

The Unity component logs JSON for clarity. A production editor should deserialize
or forward that JSON on a worker and keep the main thread for Unity objects.
WebGL cannot load this native plugin; use the browser WASM surface covered by
the next example instead.

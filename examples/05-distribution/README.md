# 05 - Consume verified distribution packages

The earlier examples build Rhythm Map from source. This example defines the
binary hand-off: a consumer downloads one platform package, verifies its exact
contents, then uses the same CLI, C ABI, or browser API without a local Rust
toolchain.

Distribution packages contain no model weights. Native packages include the
checked-in Beat This model manifest so an integrator can acquire and verify the
separately licensed artifacts explicitly. The browser package remains a
host-observation time-map engine and does not claim end-to-end beat inference.

## Package shapes

Native packages are named `rhythm-map-native-<version>-<target>` and contain:

```text
bin/rhythm-map[.exe]
include/rhythm_map.h
lib/rhythm_map_ffi.{dll,so,dylib}
models/beat-this-full-v1.json
manifest.json
SHA256SUMS
verify_package.py
```

The browser package is named
`rhythm-map-browser-wasm-<version>-wasm32-unknown-unknown`. It contains the
generated `pkg/rhythm_map.js` and `pkg/rhythm_map_bg.wasm`, TypeScript
declarations when emitted by `wasm-bindgen`, and the runnable `04` static demo.

Every `manifest.json` records the package schema, workspace version, target,
full Git commit, capabilities, entry points, and a digest for every payload
file. `SHA256SUMS` also covers the manifest itself.

## Verify before use

Python 3 is sufficient on Windows, macOS, and Linux:

```bash
python verify_package.py /path/to/unpacked-package
```

The verifier rejects path traversal, duplicate checksum entries, missing or
unlisted files, content changes, unsupported manifest schemas, and abbreviated
Git identities.

After verification, continue with [`03-c-ffi`](../03-c-ffi/) for native hosts
or [`04-browser-wasm`](../04-browser-wasm/) for browsers.

For the native CLI, acquire the default model pack once, then work offline:

```bash
bin/rhythm-map models fetch --cache-dir /path/to/model-cache
bin/rhythm-map models verify --cache-dir /path/to/model-cache
bin/rhythm-map song.mp3 --cache-dir /path/to/model-cache --output timing.json
```

Use `.exe` on Windows. `RHYTHM_MAP_CACHE_DIR` can replace the repeated cache
argument. Fetch/verify returns JSON with `model_dir`; foreign-language callers
pass that directory and `models/beat-this-full-v1.json` to the verified C ABI
constructor shown in example `03`. The CLI never downloads during analysis,
and the C ABI and WASM contain no acquisition code. See
[`models/README.md`](../../models/README.md) for cache layout and recovery.

## Build packages from a checkout

Build native binaries first, then assemble them into a fresh output directory:

```bash
cargo build --locked --release -p rhythm-map-cli -p rhythm-map-ffi
cargo run --locked -p rhythm-map-dist -- native \
  --input-dir target/release \
  --output-dir /path/to/rhythm-map-native-0.1.0-<host-target> \
  --target <host-target> \
  --git-commit <full-commit-sha>
```

The `Distribution` GitHub Actions workflow performs the native build on all
three desktop operating systems and creates the browser bundle. A manual run
uploads CI artifacts. A tag run additionally requires the tag to equal the
workspace version exactly before publishing a versioned GitHub Release whose
package manifests record the immutable source commit.

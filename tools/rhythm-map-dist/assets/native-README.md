# Rhythm Map native distribution

This package contains the `rhythm-map` command-line program and the stable C
ABI dynamic library for one target platform. C, Python, C#, Unity, and other
native hosts share the declarations in `include/rhythm_map.h`.

Model weights are not bundled. Acquire the built-in pinned pack explicitly:

```text
bin/rhythm-map models fetch
bin/rhythm-map models verify
bin/rhythm-map song.mp3 --output timing.json
```

On Windows use `bin/rhythm-map.exe`. Set `RHYTHM_MAP_CACHE_DIR` or supply
`--cache-dir` on each command to choose the cache disk. Analysis is offline;
only `models fetch` accesses the network. Existing artifacts can be used with
`--model-dir` instead. Fetch/verify prints the verified artifact directory for
the C ABI, Python, C#, and Unity integrations. Keep the accompanying model
manifest and upstream license information. Corrupt caches are rejected, never
silently overwritten. The package manifest records the exact source commit and
hashes every shipped file; model digests live in the separate model manifest.

Run `python verify_package.py .` before installing or redistributing the
package. See the repository's `examples/03-c-ffi` directory for complete C,
Python, C#, and Unity calls.

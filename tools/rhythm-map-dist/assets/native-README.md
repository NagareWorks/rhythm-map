# Rhythm Map native distribution

This package contains the `rhythm-map` command-line program and the stable C
ABI dynamic library for one target platform. C, Python, C#, Unity, and other
native hosts share the declarations in `include/rhythm_map.h`.

Model weights are not bundled. Download the files named by
`models/beat-this-full-v1.json`, retain their upstream license information, and
let Rhythm Map verify their size and SHA-256 before analysis. The package
manifest records the exact source commit and hashes every shipped file.

Run `python verify_package.py .` before installing or redistributing the
package. See the repository's `examples/03-c-ffi` directory for complete C,
Python, C#, and Unity calls.

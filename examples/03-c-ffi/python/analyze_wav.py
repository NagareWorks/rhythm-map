"""Analyze a PCM16 WAV file through the Rhythm Map C ABI."""

from __future__ import annotations

import argparse
import array
import json
import sys
import wave
from pathlib import Path

from rhythm_map import Analyzer, abi_version


def read_pcm16_wav(path: Path) -> tuple[array.array[float], int, int]:
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2 or source.getcomptype() != "NONE":
            raise ValueError("example accepts uncompressed PCM16 WAV input")
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        samples = array.array("h", source.readframes(source.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    return array.array("f", (sample / 32768.0 for sample in samples)), sample_rate, channels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--abi-only", action="store_true")
    parser.add_argument("manifest", nargs="?", type=Path)
    parser.add_argument("model_dir", nargs="?", type=Path)
    parser.add_argument("audio", nargs="?", type=Path)
    args = parser.parse_args()

    if args.abi_only:
        print(f"Rhythm Map ABI {abi_version(args.library)}")
        return
    if args.manifest is None or args.model_dir is None or args.audio is None:
        parser.error("manifest, model_dir, and audio are required unless --abi-only is used")

    samples, sample_rate, channels = read_pcm16_wav(args.audio)
    with Analyzer(args.library, args.manifest, args.model_dir) as analyzer:
        analysis = analyzer.analyze_pcm(samples, sample_rate, channels)
    print(json.dumps(analysis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

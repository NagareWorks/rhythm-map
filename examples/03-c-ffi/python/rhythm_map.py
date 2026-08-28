"""Small ctypes binding over the stable Rhythm Map C ABI."""

from __future__ import annotations

import array
import ctypes
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class RhythmMapError(RuntimeError):
    """A failure reported by the native Rhythm Map library."""


class Analyzer:
    """Own one native analyzer and expose its schema-versioned JSON result."""

    def __init__(self, library: str | os.PathLike[str], manifest: str | os.PathLike[str], model_dir: str | os.PathLike[str]) -> None:
        self._library = ctypes.CDLL(os.fspath(Path(library).resolve()))
        self._configure_abi()
        if self._library.rhythm_map_abi_version() != 1:
            raise RhythmMapError("unsupported Rhythm Map ABI version")
        self._handle = self._library.rhythm_map_analyzer_new_from_model_pack(
            os.fsencode(manifest), os.fsencode(model_dir)
        )
        if not self._handle:
            raise RhythmMapError(self._last_error())

    def _configure_abi(self) -> None:
        library = self._library
        library.rhythm_map_abi_version.argtypes = []
        library.rhythm_map_abi_version.restype = ctypes.c_uint32
        library.rhythm_map_analyzer_new_from_model_pack.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        library.rhythm_map_analyzer_new_from_model_pack.restype = ctypes.c_void_p
        library.rhythm_map_analyze_pcm_json.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint16,
        ]
        library.rhythm_map_analyze_pcm_json.restype = ctypes.c_void_p
        library.rhythm_map_last_error.argtypes = []
        library.rhythm_map_last_error.restype = ctypes.c_char_p
        library.rhythm_map_string_free.argtypes = [ctypes.c_void_p]
        library.rhythm_map_string_free.restype = None
        library.rhythm_map_analyzer_free.argtypes = [ctypes.c_void_p]
        library.rhythm_map_analyzer_free.restype = None

    def _last_error(self) -> str:
        value = self._library.rhythm_map_last_error()
        return value.decode("utf-8", errors="replace") if value else "unknown native error"

    def analyze_pcm(self, samples: Iterable[float], sample_rate: int, channels: int) -> dict[str, Any]:
        if self._handle is None:
            raise RhythmMapError("analyzer is closed")
        pcm = samples if isinstance(samples, array.array) and samples.typecode == "f" else array.array("f", samples)
        buffer_type = ctypes.c_float * len(pcm)
        buffer = buffer_type.from_buffer(pcm)
        result = self._library.rhythm_map_analyze_pcm_json(
            self._handle, buffer, len(pcm), sample_rate, channels
        )
        if not result:
            raise RhythmMapError(self._last_error())
        try:
            return json.loads(ctypes.string_at(result).decode("utf-8"))
        finally:
            self._library.rhythm_map_string_free(result)

    def close(self) -> None:
        if self._handle is not None:
            self._library.rhythm_map_analyzer_free(self._handle)
            self._handle = None

    def __enter__(self) -> Analyzer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def abi_version(library: str | os.PathLike[str]) -> int:
    """Load a native library and return its ABI version."""
    native = ctypes.CDLL(os.fspath(Path(library).resolve()))
    native.rhythm_map_abi_version.argtypes = []
    native.rhythm_map_abi_version.restype = ctypes.c_uint32
    return int(native.rhythm_map_abi_version())

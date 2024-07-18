"""Tests for energy_vad."""

import wave
from pathlib import Path
from typing import List

import pytest

from energy_vad import EnergyVad

_DIR = Path(__file__).parent


def read_wav(file_name: str) -> bytes:
    """Return audio bytes from WAV file (16Khz, 16-bit mono)."""
    with wave.open(str(_DIR / file_name), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnchannels() == 1

        return wav_file.readframes(wav_file.getnframes())


def run_vad(audio: bytes) -> List[bool]:
    """Run VAD on audio and return speech/silence for each chunk."""
    vad = EnergyVad()
    results = []
    offset = 0
    while (offset + vad.bytes_per_chunk) < len(audio):
        chunk = audio[offset : offset + vad.bytes_per_chunk]
        result = vad.process_chunk(chunk)
        if result is not None:
            results.append(result)
        offset += vad.bytes_per_chunk

    return results


def test_silence() -> None:
    """Test on WAV with no speech."""
    audio = read_wav("silence.wav")
    is_speech = run_vad(audio)
    assert all(not v for v in is_speech)


def test_speech() -> None:
    """Test on WAV with some speech."""
    audio = read_wav("speech.wav")
    is_speech = run_vad(audio)
    assert any(v for v in is_speech)


def test_bad_chunk_size() -> None:
    """Test chunk size requirement in process_chunk."""
    vad = EnergyVad()
    with pytest.raises(ValueError):
        vad.process_chunk(bytes(vad.bytes_per_chunk + 1))

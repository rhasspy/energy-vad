"""Simple energy-based voice activity detector."""

import array
import math
import statistics
from typing import List, Optional

_SAMPLE_RATE = 16000
_SAMPLE_WIDTH = 2  # 16-bit


class EnergyVad:
    """Energy-based voice activity detector (VAD)."""

    def __init__(
        self,
        threshold: Optional[float] = None,
        samples_per_chunk: int = 240,
        calibrate_seconds: float = 0.5,
    ) -> None:
        """
        Initialize VAD.

        Parameters
        ----------
        threshold: float, optional
            Energy threshold, above which is considered speech.
            If None, it will be set during calibration.
        samples_per_chunk: int
            Number of samples that process_chunk will expect.
        calibrate_seconds: float
            Seconds of audio that will be used to calibrate threshold.
        """
        self.samples_per_chunk = samples_per_chunk
        self.bytes_per_chunk = self.samples_per_chunk * _SAMPLE_WIDTH
        self.seconds_per_chunk = self.samples_per_chunk / _SAMPLE_RATE

        self.threshold = threshold

        self.calibrate_seconds = calibrate_seconds
        self._calibrate_seconds_left = self.calibrate_seconds
        self._calibrate_energies: List[float] = []

    def reset_calibration(self) -> None:
        """Reset threshold and calibration time."""
        self._calibrate_seconds_left = self.calibrate_seconds
        self._calibrate_energies.clear()
        self.threshold = None

    def process_chunk(self, chunk: bytes) -> Optional[bool]:
        """
        Process a single chunk of audio.

        Parameters
        ----------
        chunk: bytes
            Chunk of 16-bit mono PCM audio at 16Khz.
            Must be exactly self.bytes_per_chunk in size.

        Returns
        -------
        bool, optional
            None if chunk was used for calibration.
            True if energy > threshold, False otherwise.
        """
        if len(chunk) != self.bytes_per_chunk:
            raise ValueError(f"Chunk must be {self.bytes_per_chunk} bytes")

        chunk_array = array.array("h", chunk)

        # Compute RMS
        energy = -math.sqrt(sum(x**2 for x in chunk_array) / len(chunk_array))
        debiased_energy = math.sqrt(
            sum((x + energy) ** 2 for x in chunk_array) / len(chunk_array)
        )

        if self.threshold is None:
            if self._calibrate_seconds_left <= 0:
                # Calibration complete
                self.threshold = statistics.mean(self._calibrate_energies) + math.sqrt(
                    statistics.variance(self._calibrate_energies)
                )
            else:
                self._calibrate_energies.append(debiased_energy)
                self._calibrate_seconds_left -= self.seconds_per_chunk

            # Calibrating
            return None

        return debiased_energy > self.threshold

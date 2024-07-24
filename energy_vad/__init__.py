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
        calibrate_zscore_threshold: float = 1.0,
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
        calibrate_zscore_threshold: float
            Only energies below this (median) z-score threshold will be used in
            calibration.
        """
        self.samples_per_chunk = samples_per_chunk
        self.bytes_per_chunk = self.samples_per_chunk * _SAMPLE_WIDTH
        self.seconds_per_chunk = self.samples_per_chunk / _SAMPLE_RATE

        self.threshold = threshold

        self.calibrate_seconds = calibrate_seconds
        self.calibrate_zscore_threshold = calibrate_zscore_threshold
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
                # Enough energy values are available for calibration.
                # Calculate median z-score to remove high-energy clicks.
                median = statistics.median(self._calibrate_energies)
                stdev = statistics.stdev(self._calibrate_energies)
                z_score = [(x - median) / stdev for x in self._calibrate_energies]

                # Filter outliers, but fall back to using all energies if
                # everything gets filtered out.
                energies = [
                    x
                    for i, x in enumerate(self._calibrate_energies)
                    if z_score[i] < self.calibrate_zscore_threshold
                ] or self._calibrate_energies

                self.threshold = statistics.mean(energies) + statistics.stdev(energies)
            else:
                self._calibrate_energies.append(debiased_energy)
                self._calibrate_seconds_left -= self.seconds_per_chunk

            # Calibrating
            return None

        return debiased_energy > self.threshold

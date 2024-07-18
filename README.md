# Energy VAD

Simple energy-based voice activity detector (VAD) with no external dependencies.

Energy threshold is calibrated from initial audio, or can be set manually.

## Installation

``` sh
pip install energy-vad
```

## Example

``` python
import wave
from energy_vad import EnergyVad

vad = EnergyVad()

with wave.open("test.wav", "rb") as wav_file:
    assert wav_file.getframerate() == 16000
    assert wav_file.getsampwidth() == 2
    assert wav_file.getnchannels() == 1
    
    chunk = wav_file.readframes(vad.samples_per_chunk)
    while len(chunk) == vad.bytes_per_chunk:
        result = vad.process_chunk(chunk)
        if result is None:
            # calibrating
            pass
        elif result:
            # speech
            print("!", end="")
        else:
            # silence
            print(".", end="")

        chunk = wav_file.readframes(vad.samples_per_chunk)
        
print("")
print("Energy threshold:", vad.threshold)

# Clear calibrated threshold
vad.reset_calibration()
```


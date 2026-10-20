"""Play a WAV file through the default output device (simulates remote participants for testing).

    .venv\\Scripts\\python.exe scripts\\play_wav.py tests\\data\\two_speakers_16k.wav [--gain 0.8] [--device NAME]

The app's WASAPI loopback capture then hears it exactly like a meeting.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--gain", type=float, default=0.8)
    ap.add_argument("--device", help="output device name substring (default: default output)")
    ap.add_argument("--lead-silence", type=float, default=0.5)
    args = ap.parse_args()

    import pyaudiowpatch as pyaudio

    audio, sr = sf.read(args.wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.clip(audio * args.gain, -1.0, 1.0)

    p = pyaudio.PyAudio()
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        dev = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        if args.device:
            for i in range(p.get_device_count()):
                d = p.get_device_info_by_index(i)
                if d["hostApi"] == wasapi["index"] and d["maxOutputChannels"] > 0 and args.device.lower() in d["name"].lower():
                    dev = d
                    break
        rate = int(dev["defaultSampleRate"])
        ch = min(2, int(dev["maxOutputChannels"]))
        # resample to the device rate (linear; fine for a test signal)
        if rate != sr:
            n = int(len(audio) * rate / sr)
            audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
        lead = np.zeros(int(args.lead_silence * rate), dtype=np.float32)
        mono = np.concatenate([lead, audio, np.zeros(rate // 2, dtype=np.float32)])
        frames = np.repeat(mono[:, None], ch, axis=1).astype(np.float32).tobytes()
        print(f"playing {args.wav} on [{dev['index']}] {dev['name']} @ {rate} Hz x{ch}, {len(mono) / rate:.1f} s")
        stream = p.open(format=pyaudio.paFloat32, channels=ch, rate=rate, output=True, output_device_index=dev["index"])
        t0 = time.time()
        stream.write(frames)
        stream.stop_stream()
        stream.close()
        print(f"done in {time.time() - t0:.1f} s")
    finally:
        p.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())

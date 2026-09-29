"""Synthesize the played-notes MIDI into a piano-ish tone and mux it with the
rollout video: `python scripts/mj/midi_to_audio_video.py played.mid in.mp4 out.mp4`.
No fluidsynth needed: additive synth (decaying harmonics) per note."""
import sys, subprocess, numpy as np, pretty_midi
from scipy.io import wavfile

mid, vid, out = sys.argv[1:4]
fs = 44100
pm = pretty_midi.PrettyMIDI(mid)
T = pm.get_end_time() + 1.5
audio = np.zeros(int(T * fs), dtype=np.float64)
harm = [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12), (5, 0.06)]
for inst in pm.instruments:
    for n in inst.notes:
        f0 = pretty_midi.note_number_to_hz(n.pitch)
        dur = max(n.end - n.start, 0.05) + 0.4            # let it ring a bit after release
        t = np.arange(int(dur * fs)) / fs
        env = np.exp(-3.0 * t) * (1 - np.exp(-t * 400))    # fast attack, exponential decay
        # cut the sustain once the key is released (with a short tail)
        rel = n.end - n.start
        env *= np.where(t < rel, 1.0, np.exp(-(t - rel) * 12.0))
        wave = sum(a * np.sin(2 * np.pi * f0 * k * t) for k, a in harm)
        s = int(n.start * fs); e = min(s + len(t), len(audio))
        audio[s:e] += (n.velocity / 127.0) * env[:e - s] * wave[:e - s]
audio /= max(np.abs(audio).max(), 1e-6) * 1.1
wav = out.rsplit(".", 1)[0] + ".wav"
wavfile.write(wav, fs, (audio * 32767).astype(np.int16))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", vid, "-i", wav,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=True)
print(f"[audio] {len(pm.instruments[0].notes)} notes -> {wav}\n[video] -> {out}")

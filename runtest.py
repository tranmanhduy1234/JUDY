import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

# Các mức samplerate để so sánh
samplerates = [8000, 16000, 44100]
duration = 2  # số giây ghi âm

recordings = {}

print("🎙️ Hãy nói trong 2 giây...")

# Ghi âm ở từng mức samplerate
for sr in samplerates:
    print(f"Đang ghi với samplerate = {sr}Hz ...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    recordings[sr] = audio.flatten()

# Vẽ biểu đồ riêng cho từng samplerate
fig, axes = plt.subplots(len(samplerates), 1, figsize=(12, 8))

for idx, sr in enumerate(samplerates):
    audio = recordings[sr]
    time_axis = np.linspace(0, duration, len(audio))
    axes[idx].plot(time_axis, audio)
    axes[idx].set_title(f"Waveform ở samplerate = {sr} Hz")
    axes[idx].set_xlabel("Thời gian (s)")
    axes[idx].set_ylabel("Biên độ")
    axes[idx].grid(True)

plt.tight_layout()
plt.show()
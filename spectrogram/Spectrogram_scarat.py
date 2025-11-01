import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 1️⃣ Đọc file âm thanh
samplerate, data = wavfile.read("D:\ptithcm\HTTM\Driver drowsiness audio\data_reallife\yes2.wav")

# Chuyển sang mono nếu là stereo
if len(data.shape) == 2:
    data = data.mean(axis=1)

# 2️⃣ Chuẩn bị các thông số cho spectrogram
window_size = 1024        # số mẫu mỗi cửa sổ
hop_size = 512            # bước nhảy giữa các cửa sổ
window = np.hanning(window_size)  # Hanning window để giảm rìa

# 3️⃣ Chia tín hiệu thành các frame
frames = []
for start in range(0, len(data) - window_size, hop_size):
    frame = data[start:start + window_size] * window
    frames.append(frame)
frames = np.array(frames)

# 4️⃣ Tính FFT cho từng frame
fft_size = window_size
spectrogram = np.abs(np.fft.rfft(frames, n=fft_size))  # magnitude
spectrogram = 20 * np.log10(spectrogram + 1e-8)        # dB scale

# 5️⃣ Tạo trục thời gian và tần số
times = np.arange(frames.shape[0]) * hop_size / samplerate
frequencies = np.fft.rfftfreq(fft_size, 1 / samplerate)

# 6️⃣ Vẽ spectrogram
plt.figure(figsize=(10, 6))
plt.pcolormesh(times, frequencies, spectrogram.T, shading='gouraud', cmap='magma')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title('Spectrogram (manual implementation)')
plt.colorbar(label='Magnitude [dB]')
plt.savefig("spectrogram//yes2.png", dpi=300, bbox_inches='tight')
plt.show()
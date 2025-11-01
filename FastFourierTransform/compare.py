import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

# Đọc file wav
fs, signal = wav.read("FastFourierTransform//first_recording.wav")

# Nếu stereo → lấy 1 kênh
if len(signal.shape) > 1:
    signal = signal[:,0]

# Chọn một đoạn tín hiệu (ví dụ 0.05 giây)
N = int(5 * fs)
segment = signal[:N]

# FFT
fft_result = np.fft.fft(segment)
freqs = np.fft.fftfreq(N, 1/fs)
half = N//2

fft_magnitude = np.abs(fft_result[:half])
freqs_half = freqs[:half]

# Vẽ so sánh
plt.figure(figsize=(12,5))

# Miền thời gian
time_axis = np.arange(N) / fs
plt.subplot(1,2,1)
plt.plot(time_axis, segment)
plt.title("Tín hiệu gốc (Miền thời gian)")
plt.xlabel("Thời gian (s)")
plt.ylabel("Biên độ")

# Miền tần số
plt.subplot(1,2,2)
plt.semilogy(freqs_half, fft_magnitude)  # log scale để dễ nhìn
plt.title("Sau biến đổi Fourier (Miền tần số)")
plt.xlabel("Tần số (Hz)")
plt.ylabel("Biên độ (log)")

plt.tight_layout()
plt.show()

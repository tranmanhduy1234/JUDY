import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

# Đọc file wav
fs, signal = wav.read("FastFourierTransform//first_recording.wav")

# Nếu stereo → lấy 1 kênh
if len(signal.shape) > 1:
    signal = signal[:,0]

# Chọn một đoạn (ví dụ 4096 mẫu đầu)
N = 4096
segment = signal

# FFT
fft_result = np.fft.fft(segment)
freqs = np.fft.fftfreq(N, 1/fs)
half = N//2

fft_magnitude = np.abs(fft_result[:half])
freqs_half = freqs[:half]

# Vẽ
plt.figure(figsize=(10,5))
plt.semilogy(freqs_half, fft_magnitude)  # log-scale trên trục Y
plt.title("Phổ tần số (FFT - log scale)")
plt.xlabel("Tần số (Hz)")
plt.ylabel("Biên độ (log)")
plt.grid()
plt.show()


from scipy.signal import spectrogram

f, t, Sxx = spectrogram(signal, fs)

plt.figure(figsize=(10,5))
plt.pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud')
plt.title("Spectrogram")
plt.ylabel("Tần số (Hz)")
plt.xlabel("Thời gian (s)")
plt.colorbar(label="Cường độ (dB)")
plt.ylim(0, 4000)  # cắt ở 4kHz cho dễ nhìn, có thể bỏ nếu muốn full
plt.show()
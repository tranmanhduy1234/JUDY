import torch
import torchaudio
import matplotlib.pyplot as plt

waveform, sr = torchaudio.load(r"D:\ptithcm\HTTM\Driver drowsiness audio\data_reallife\yes2.wav")
spec = torchaudio.transforms.Spectrogram()(waveform)
import random
def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = (sr // 1000) * max_ms
        
        if max_len < sig_len:
            sig = sig[:,:max_len] # cắt âm thanh
        elif max_len > sig_len:
            pad_beggin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_beggin_len
            
            pad_beggin = torch.zeros((num_rows, pad_beggin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            sig = torch.cat((pad_beggin, sig, pad_end), dim=1)
        return (sig, sr)
waveform, sr = pad_trunc((waveform, sr), 1000)
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr,
    n_fft=1024,
    hop_length=512,
    n_mels=128
)(waveform)

plt.imshow(mel_spec.log2()[0,:,:].numpy(), cmap='magma')
plt.title("")
plt.xlabel("Time")
plt.ylabel("Mel Frequency")
plt.colorbar()
plt.savefig("spectrogram//yes2.png", dpi=300, bbox_inches='tight')
plt.show()
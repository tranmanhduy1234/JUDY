import sounddevice as sd
import numpy as np
import whisper.whisper as whisper
import tempfile
import scipy.io.wavfile as wav
import os

# Load Whisper model (tiny/ base/ small/ medium/ large)
model = whisper.load_model("small")

samplerate = 16000
duration = 5  # số giây cho mỗi lần nghe

saved_first_file = False  # Biến đánh dấu đã lưu file đầu tiên chưa
first_file_path = "first_recording.wav"  # File lưu lâu dài

while True:
    print("🎙️ Nói đi (bấm Ctrl+C để thoát)...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()

    if not saved_first_file:
        # Lưu file đầu tiên lâu dài
        wav.write(first_file_path, samplerate, recording)
        saved_first_file = True
        file_to_transcribe = first_file_path
        print(f"✅ File đầu tiên được lưu lâu dài: {first_file_path}")
    else:
        # Các file khác vẫn lưu tạm
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav.write(f.name, samplerate, recording)
            file_to_transcribe = f.name

    # Nhận diện bằng Whisper
    result = model.transcribe(file_to_transcribe, language="vi")
    print("📝 Bạn nói:", result["text"])
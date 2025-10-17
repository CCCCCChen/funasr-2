import librosa
import io

def process(audio_bytes: bytes):
    # 加载并重采样到 16kHz
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    duration = len(y) / sr
    return [{"audio": y, "sample_rate": sr, "duration": duration}]

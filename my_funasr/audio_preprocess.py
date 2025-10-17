import librosa
import io

# 现有的简单处理：返回统一采样率的单段
def process(audio_bytes: bytes):
    # 加载并重采样到 16kHz
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
    duration = len(y) / sr
    return [{"audio": y, "sample_rate": sr, "duration": duration}]

# 新增：管线使用的加载函数，返回 (波形, 采样率, 时长)
def load_audio(audio_bytes: bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
    duration = len(y) / sr
    return y, sr, duration

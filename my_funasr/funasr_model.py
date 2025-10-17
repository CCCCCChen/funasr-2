from funasr import AutoModel
import time
import logging

logger = logging.getLogger("funasr")

_model = None

def load_model(model_dir: str, device="cpu"):
    global _model
    _model = AutoModel(model=model_dir, device=device)
    try:
        model_path = getattr(_model, "model_path", None) or model_dir
        logger.info(f"FunASR model loaded: {model_path} on {device}")
    except Exception:
        pass
    return _model

def infer(audio, sample_rate=16000):
    global _model
    t0 = time.perf_counter()
    res = _model.generate(input=audio, batch_size=1)
    elapsed = time.perf_counter() - t0
    try:
        audio_dur = (len(audio) / float(sample_rate)) if sample_rate else 0.0
    except Exception:
        audio_dur = 0.0
    rtf = (elapsed / audio_dur) if audio_dur > 0 else None
    # 输出是一个列表，取第一个结果
    out = res[0]
    # 附加一些元信息，便于日志与排查
    try:
        out.setdefault("_meta", {})
        out["_meta"].update({
            "elapsed_s": elapsed,
            "audio_dur_s": audio_dur,
            "rtf": rtf,
            "sr": sample_rate,
            "n_samples": len(audio),
        })
    except Exception:
        pass
    try:
        if rtf is not None:
            logger.info(f"Infer done: dur={audio_dur:.3f}s time={elapsed:.3f}s rtf={rtf:.3f}")
        else:
            logger.info(f"Infer done: dur={audio_dur:.3f}s time={elapsed:.3f}s")
    except Exception:
        pass
    return out

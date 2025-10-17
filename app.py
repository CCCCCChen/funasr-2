from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from my_funasr import funasr_model, audio_preprocess, text_postprocess
import logging

logger = logging.getLogger("asr")

app = FastAPI(title="FunASR Service", description="Offline ASR API for Dify", version="1.0.0")

@app.on_event("startup")
def load_funasr():
    global model
    model = funasr_model.load_model(
        # model_dir="models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        model_dir="models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        device="cpu"  # 改成 "cuda:0" 也可以
    )
    print("✅ FunASR model loaded successfully!")

class ASRResponse(BaseModel):
    text: str
    segments: Optional[List[Dict[str, Any]]] = None
    duration: Optional[float] = None

@app.post("/asr", response_model=ASRResponse)
async def asr_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ["audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3"]:
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    audio_bytes = await file.read()
    try:
        logger.info(f"Incoming audio: type={file.content_type} size={len(audio_bytes)} bytes")
    except Exception:
        pass

    segments = audio_preprocess.process(audio_bytes)
    if not segments:
        return ASRResponse(text="", segments=[], duration=0.0)

    total_seg_dur = sum(s["duration"] for s in segments)
    try:
        logger.info(f"Preprocess: segments={len(segments)} total_dur={total_seg_dur:.3f}s")
    except Exception:
        pass

    results = []
    for i, seg in enumerate(segments, 1):
        try:
            logger.info(f"Infer seg {i}/{len(segments)}: sr={seg['sample_rate']} dur={seg['duration']:.3f}s samples={len(seg['audio'])}")
        except Exception:
            pass
        # 调用我们封装的模块级 infer，而不是 AutoModel 的方法
        output = funasr_model.infer(seg["audio"], seg["sample_rate"])
        results.append({"output": output, "duration": seg["duration"]})

    text, segs = text_postprocess.combine(results)
    total_dur = sum(s["duration"] for s in segments)
    try:
        logger.info(f"Postprocess: text_len={len(text)} total_dur={total_dur:.3f}s")
    except Exception:
        pass

    return ASRResponse(text=text, segments=segs, duration=total_dur)

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8220)

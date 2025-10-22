from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from my_funasr.funasr_pipeline import FunASRPipeline
from task_manager import TaskManager
import uuid, asyncio
import os
import torch

# 设备自动检测：优先 MPS（Apple GPU），其次 CUDA，最后 CPU
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
torch.set_float32_matmul_precision("high")

def detect_device():
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

device = detect_device()

app = FastAPI(title="FunASR Async Service", description="ASR with punctuation & diarization", version="2.0")

# 初始化任务管理器与模型管线
task_manager = TaskManager()
asr_pipeline = FunASRPipeline(
    # 主线 ASR 改为 SenseVoiceSmall
    asr_model_dir="/Users/minichen/Downloads/Models/funasr-python/funasr-2/models/SenseVoiceSmall",
    # 可选：外部标点模型（一般不需要）
    punc_model_dir="/Users/minichen/Downloads/Models/funasr-python/funasr-2/models/punc_ct-transformer_cn-en-common-vocab471067-large",
    # 可选：说话人分离/标注模型（当前为 SV，默认禁用分离）
    spk_model_dir="/Users/minichen/Downloads/Models/funasr-python/funasr-2/models/speech_campplus_sv_zh-cn_16k-common",
    device=device,
    use_external_punc=False,
    use_diarization=False,
    # 关闭独立 SenseSmallVoice，避免重复加载
    sense_model_dir=None,
    # 官方 VAD 暂不启用
    vad_model_dir=None,
    # 增强模型使用 ModelScope ID，确保加载正确的管线
    enhance_model_dir="dengcunqin/speech_mossformer2_noise_reduction_16k",
)

@app.get("/env")
def env():
    return {
        "device": device,
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "matmul_precision": "high",
    }

@app.post("/asr/submit")
async def submit_asr(file: UploadFile = File(...)):
    """接收音频文件并启动异步识别任务（增强->VAD分段/分离->ASR）"""
    if file.content_type not in ["audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3"]:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    audio_bytes = await file.read()
    task_id = str(uuid.uuid4())

    # 注册任务 + 缓存原始音频
    task_manager.create_task(task_id, status="pending", payload={"audio_bytes": audio_bytes})

    # 启动后台任务（全流程）
    asyncio.create_task(asr_pipeline.run_full_task(task_id, audio_bytes, task_manager))

    return {"task_id": task_id, "status": "processing"}

@app.post("/asr/enhanced/{task_id}")
async def run_enhanced(task_id: str):
    """仅运行增强模型，结果写入 task.stages.enhanced"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    asyncio.create_task(asr_pipeline.run_enhanced_task(task_id, task_manager))
    return {"task_id": task_id, "stage": "enhanced", "status": "processing"}

@app.post("/asr/diarization/{task_id}")
async def run_diarization(task_id: str):
    """仅运行官方 VAD 分段，结果写入 task.stages.diarization"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    asyncio.create_task(asr_pipeline.run_vad_task(task_id, task_manager))
    return {"task_id": task_id, "stage": "diarization", "status": "processing"}

@app.post("/asr/transformer/{task_id}")
async def run_transformer(task_id: str):
    """使用 SenseSmallVoice 进行识别，结果写入 task.stages.transformer"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    asyncio.create_task(asr_pipeline.run_transformer_task(task_id, task_manager))
    return {"task_id": task_id, "stage": "transformer", "status": "processing"}

@app.get("/asr/status/{task_id}")
def check_status(task_id: str):
    """查询任务状态：返回正在进行的进度与信息，完成时返回结果，错误时返回错误详情"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    status = task.get("status")
    if status == "done":
        return {
            "status": "done",
            "result": task.get("result"),
            "stages": task.get("stages")
        }
    elif status == "error":
        return {
            "status": "error",
            "error": task.get("error"),
            "message": task.get("message"),
            "stages": task.get("stages")
        }
    else:
        return {
            "status": status,
            "progress": task.get("progress"),
            "message": task.get("message"),
            "stages": task.get("stages")
        }

@app.get("/health")
def health():
    return {"status": "ok"}

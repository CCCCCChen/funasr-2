from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from my_funasr.funasr_pipeline import FunASRPipeline
from task_manager import TaskManager
import uuid, asyncio

from modelscope import snapshot_download
from pathlib import Path
import torch
import shutil
import hashlib

# 下载/定位模型到项目 models 目录
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
model_dir = snapshot_download(
    'iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    cache_dir=str(MODELS_DIR)
)

app = FastAPI(title="FunASR Async Service", description="ASR with punctuation & diarization", version="2.0")

# 初始化任务管理器与模型管线
task_manager = TaskManager()

# 自动选择设备：若CUDA不可用则使用CPU，避免 Torch not compiled with CUDA enabled
device = "cuda:0" if torch.cuda.is_available() else "cpu"
asr_pipeline = FunASRPipeline(
    asr_model_dir=model_dir,  # 使用本地已下载模型路径
    punc_model_dir="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    spk_model_dir="iic/speech_campplus_sv_zh-cn_16k-common",
    device=device,
    use_external_punc=False,
    use_diarization=False
)

@app.post("/asr/submit")
async def submit_asr(file: UploadFile = File(...)):
    """接收音频文件并启动异步识别任务（支持大文件与 m4a），按块落盘并计算哈希"""
    # 落盘到 uploads 目录，并计算 sha256 哈希
    uploads_dir = BASE_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    orig_name = file.filename or "upload"
    suffix = Path(orig_name).suffix or ""
    temp_path = uploads_dir / f"{uuid.uuid4()}{suffix}"

    hasher = hashlib.sha256()
    try:
        with temp_path.open("wb") as f:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                hasher.update(chunk)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to persist uploaded file")
    file_hash = hasher.hexdigest()

    task_id = str(uuid.uuid4())

    # 注册任务（记录文件哈希）
    task_manager.create_task(task_id, status="pending", file_hash=file_hash)

    # 启动后台任务：传递文件路径以便后端解码
    asyncio.create_task(asr_pipeline.run_task(task_id, str(temp_path), task_manager))

    return {"task_id": task_id, "file_hash": file_hash, "status": "processing"}

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
            "result": task.get("result")
        }
    elif status == "error":
        return {
            "status": "error",
            "error": task.get("error"),
            "message": task.get("message")
        }
    else:
        return {
            "status": status,
            "progress": task.get("progress"),
            "message": task.get("message")
        }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/asr/status_by_hash/{file_hash}")
def check_status_by_hash(file_hash: str):
    """按文件哈希查询最新任务状态"""
    task = task_manager.get_task_by_hash(file_hash)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found for given hash")
    status = task.get("status")
    if status == "done":
        return {"status": "done", "result": task.get("result")}
    elif status == "error":
        return {"status": "error", "error": task.get("error"), "message": task.get("message")}
    else:
        return {"status": status, "progress": task.get("progress"), "message": task.get("message")}

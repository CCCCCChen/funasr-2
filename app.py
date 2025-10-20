from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from my_funasr.funasr_pipeline import FunASRPipeline
from task_manager import TaskManager
import uuid, asyncio

app = FastAPI(title="FunASR Async Service", description="ASR with punctuation & diarization", version="2.0")

# 初始化任务管理器与模型管线
task_manager = TaskManager()
asr_pipeline = FunASRPipeline(
    # asr_model_dir="models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    # punc_model_dir="models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    # spk_model_dir="models/speech_campplus_speaker-diarization-pytorch",
    # device="cuda:0"  # 改为 "cpu" 亦可
    asr_model_dir="models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    punc_model_dir="models/punc_ct-transformer_cn-en-common-vocab471067-large",
    spk_model_dir="models/speech_campplus_sv_zh-cn_16k-common",
    device="cpu",  # 改为 "cpu" 亦可
    use_external_punc=False,
    use_diarization=False
)

@app.post("/asr/submit")
async def submit_asr(file: UploadFile = File(...)):
    """接收音频文件并启动异步识别任务"""
    if file.content_type not in ["audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3"]:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    audio_bytes = await file.read()
    task_id = str(uuid.uuid4())

    # 注册任务
    task_manager.create_task(task_id, status="pending")

    # 启动后台任务
    asyncio.create_task(asr_pipeline.run_task(task_id, audio_bytes, task_manager))

    return {"task_id": task_id, "status": "processing"}

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

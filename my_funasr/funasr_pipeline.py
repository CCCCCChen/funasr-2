import io, librosa
import asyncio
from funasr import AutoModel
from my_funasr.audio_preprocess import load_audio
from my_funasr.text_postprocess import combine_segments
import os
from pathlib import Path
import torch
try:
    from modelscope import snapshot_download
except Exception:
    snapshot_download = None

class FunASRPipeline:
    def __init__(self, asr_model_dir, punc_model_dir=None, spk_model_dir=None, device="cpu", use_external_punc=False, use_diarization=None):
        def _resolve_device(dev: str):
            # 当用户传入 CUDA 但当前 PyTorch 不支持时自动回退 CPU
            if isinstance(dev, str) and dev.lower().startswith("cuda") and not torch.cuda.is_available():
                print("⚠️ Torch 未启用 CUDA，设备自动回退到 CPU")
                return "cpu"
            return dev

        def _ensure_local_model(m):
            if m is None:
                return None
            m_str = str(m)
            p = Path(m_str)
            if p.exists() and p.is_dir():
                return m_str
            # 不是本地路径，尝试拉取到项目 models 目录
            try:
                if snapshot_download is None:
                    return m_str
                project_root = Path(__file__).resolve().parent.parent  # funasr-2 根目录
                cache_dir = project_root / "models"
                cache_dir.mkdir(parents=True, exist_ok=True)
                local_dir = snapshot_download(m_str, cache_dir=str(cache_dir))
                return local_dir
            except Exception as e:
                print(f"⚠️ 模型下载失败，回退使用远程仓库默认缓存: {e}")
                return m_str

        print("🚀 Loading ASR model...")
        resolved_device = _resolve_device(device)
        asr_spec = _ensure_local_model(asr_model_dir)
        self.asr_model = AutoModel(model=asr_spec, device=resolved_device)
        self.use_external_punc = use_external_punc
        # 根据目录名自动判断是否是分离模型；若传入 False 则禁用
        self.use_diarization = use_diarization if use_diarization is not None else (
            spk_model_dir is not None and ("diar" in str(spk_model_dir) or "speaker-diarization" in str(spk_model_dir))
        )

        # 针对内存的保守配置：长音频分块 & generate 内部分批
        self.max_chunk_seconds = 60  # 每段最长 60 秒，避免一次性处理超长音频占用过多内存
        self.batch_size_seconds = 30  # generate 的 batch_size_s，按秒切分内部批次

        if self.use_external_punc and punc_model_dir:
            print("🔤 Loading punctuation model...")
            punc_spec = _ensure_local_model(punc_model_dir)
            self.punc_model = AutoModel(model=punc_spec, device=resolved_device)
        else:
            self.punc_model = None

        if self.use_diarization and spk_model_dir:
            print("🗣️ Loading diarization model...")
            spk_spec = _ensure_local_model(spk_model_dir)
            self.spk_model = AutoModel(model=spk_spec, device=resolved_device)
        else:
            self.spk_model = None
        print("✅ Models loaded.")

    def _run_task_sync(self, task_id, audio_bytes, task_manager):
        try:
            task_manager.update_task(task_id, status="running", progress=0.05, message="loading audio")

            # 加载音频
            audio, sr, duration = load_audio(audio_bytes)
            task_manager.update_task(task_id, progress=0.10, message=f"audio loaded: sr={sr}, dur={duration:.2f}s")

            # 说话人分离（仅在具备真实分离模型时启用）
            segments = []
            if self.use_diarization and self.spk_model is not None:
                try:
                    diar_result = self.spk_model.generate(input=audio, sample_rate=sr)
                    segments = diar_result.get("segments", []) if isinstance(diar_result, dict) else []
                except Exception as e:
                    segments = []
                    task_manager.update_task(task_id, message=f"diarization failed: {e}")
                if segments:
                    task_manager.update_task(task_id, progress=0.30, message=f"diarization ready: {len(segments)} segments")
                else:
                    segments = [{"start": 0.0, "end": duration, "spk": "spk0"}]
                    task_manager.update_task(task_id, progress=0.30, message="diarization empty; using full audio")
            else:
                # 未启用分离时，默认整段识别；若音频过长，进行固定时长分块
                task_manager.update_task(task_id, progress=0.30, message="diarization disabled; using full audio or chunks")
                if duration <= self.max_chunk_seconds:
                    segments = [{"start": 0.0, "end": duration, "spk": "spk0"}]
                else:
                    # 以 max_chunk_seconds 为单位进行切块
                    segments = []
                    start = 0.0
                    while start < duration:
                        end = min(start + self.max_chunk_seconds, duration)
                        segments.append({"start": start, "end": end, "spk": "spk0"})
                        start = end

            # 分段识别（当禁用分离时即分块识别）
            results = []
            total = len(segments)
            for i, seg in enumerate(segments):
                seg_audio = audio[int(seg["start"]*sr): int(seg["end"]*sr)]
                # 使用 batch_size_s 降低内存占用（单位为秒）
                try:
                    asr_out = self.asr_model.generate(input=seg_audio, sample_rate=sr, batch_size_s=self.batch_size_seconds)
                except Exception as e:
                    # 如果 batch_size_s 不被当前模型支持，则回退不传该参数
                    task_manager.update_task(task_id, message=f"asr generate warn: {e}; retry without batch_size_s")
                    asr_out = self.asr_model.generate(input=seg_audio, sample_rate=sr)

                text = asr_out[0]["text"] if asr_out and len(asr_out) else ""

                # 标点恢复：仅在明确使用外部标点模型时启用；默认使用 Paraformer 内置标点
                if self.use_external_punc and self.punc_model is not None:
                    try:
                        punc_out = self.punc_model.generate(input=text)
                        final_text = punc_out[0]["text"] if punc_out and len(punc_out) else text
                    except Exception:
                        final_text = text
                else:
                    final_text = text

                results.append({
                    "speaker": seg.get("spk", "spk" + str(i)),
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": final_text
                })

                # 进度更新：30% -> 90%
                progress = 0.30 + 0.60 * ((i + 1) / total)
                task_manager.update_task(task_id, progress=progress, message=f"segment {i+1}/{total} done")

            combined_text = combine_segments(results)
            task_manager.update_task(task_id, progress=0.95, message="postprocess combining")

            task_manager.update_task(task_id, status="done", result={
                "text": combined_text,
                "segments": results,
                "duration": duration
            }, progress=1.0, message="done")
        except Exception as e:
            task_manager.update_task(task_id, status="error", error=str(e), message="pipeline error")

    async def run_task(self, task_id, audio_bytes, task_manager):
        await asyncio.to_thread(self._run_task_sync, task_id, audio_bytes, task_manager)

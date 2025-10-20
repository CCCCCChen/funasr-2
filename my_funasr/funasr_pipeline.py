import io, librosa
import asyncio
from funasr import AutoModel
from my_funasr.audio_preprocess import load_audio
from my_funasr.text_postprocess import combine_segments

class FunASRPipeline:
    def __init__(self, asr_model_dir, punc_model_dir=None, spk_model_dir=None, device="cpu", use_external_punc=False, use_diarization=None):
        print("🚀 Loading ASR model...")
        self.asr_model = AutoModel(model=asr_model_dir, device=device)
        self.use_external_punc = use_external_punc
        # 根据目录名自动判断是否是分离模型；若传入 False 则禁用
        self.use_diarization = use_diarization if use_diarization is not None else (
            spk_model_dir is not None and ("diar" in spk_model_dir or "speaker-diarization" in spk_model_dir)
        )

        if self.use_external_punc and punc_model_dir:
            print("🔤 Loading punctuation model...")
            self.punc_model = AutoModel(model=punc_model_dir, device=device)
        else:
            self.punc_model = None

        if self.use_diarization and spk_model_dir:
            print("🗣️ Loading diarization model...")
            self.spk_model = AutoModel(model=spk_model_dir, device=device)
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
                task_manager.update_task(task_id, progress=0.30, message="diarization disabled; using full audio")
                segments = [{"start": 0.0, "end": duration, "spk": "spk0"}]

            # 分段识别（当禁用分离时即整段识别）
            results = []
            total = len(segments)
            for i, seg in enumerate(segments):
                seg_audio = audio[int(seg["start"]*sr): int(seg["end"]*sr)]
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

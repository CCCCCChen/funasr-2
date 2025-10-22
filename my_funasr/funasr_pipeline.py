import io, librosa
import asyncio
import numpy as np
from funasr import AutoModel
from my_funasr.audio_preprocess import load_audio
from my_funasr.text_postprocess import combine_segments

class FunASRPipeline:
    def __init__(self, asr_model_dir, punc_model_dir=None, spk_model_dir=None, device="cpu", use_external_punc=False, use_diarization=None,
                 sense_model_dir=None, vad_model_dir=None, enhance_model_dir=None):
        print("🚀 Loading ASR model...")
        # 设备处理：优先尝试使用传入设备（包括 mps），失败则回退到 cpu
        def _load_with_device(model_dir):
            try:
                return AutoModel(model=model_dir, device=device)
            except Exception as e:
                if device == "mps":
                    print(f"[device] MPS init failed for {model_dir}, fallback to CPU: {e}")
                    return AutoModel(model=model_dir, device="cpu")
                raise

        self.asr_model = _load_with_device(asr_model_dir)
        self.use_external_punc = use_external_punc
        # 根据目录名自动判断是否是分离模型；若传入 False 则禁用
        self.use_diarization = use_diarization if use_diarization is not None else (
            spk_model_dir is not None and ("diar" in spk_model_dir or "speaker-diarization" in spk_model_dir)
        )

        # 可选模型：SenseSmallVoice（替代 ASR 的 transformer 路线）、官方 VAD、增强
        self.sense_model = None
        if sense_model_dir:
            try:
                self.sense_model = _load_with_device(sense_model_dir)
            except Exception as e:
                print(f"[transformer] skip loading SenseSmallVoice: {e}")
        self.vad_model = None
        if vad_model_dir:
            try:
                self.vad_model = _load_with_device(vad_model_dir)
            except Exception as e:
                print(f"[vad] skip loading official VAD: {e}")
        self.enhance_model = None
        if enhance_model_dir:
            try:
                self.enhance_model = _load_with_device(enhance_model_dir)
            except Exception as e:
                print(f"[enhance] skip loading mossformer2: {e}")

        if self.use_external_punc and punc_model_dir:
            print("🔤 Loading punctuation model...")
            self.punc_model = _load_with_device(punc_model_dir)
        else:
            self.punc_model = None

        if self.use_diarization and spk_model_dir:
            print("🗣️ Loading diarization model...")
            self.spk_model = _load_with_device(spk_model_dir)
        else:
            self.spk_model = None
        print("✅ Models loaded.")

        # MPS 设备预热：短音频触发内核编译，减少首次推理延迟
        if device == "mps":
            try:
                warm = np.zeros(8000, dtype=np.float32)  # 0.5s @16k
                self.asr_model.generate(input=warm, sample_rate=16000)
                if self.enhance_model:
                    self.enhance_model.generate(input=warm, sample_rate=16000)
                print("⚡ MPS warm-up done")
            except Exception as e:
                print(f"[mps] warm-up failed: {e}")

    # ===== 单阶段：增强 =====
    def _enhanced_sync(self, task_id, task_manager):
        try:
            task_manager.update_task(task_id, status="running", progress=0.05, message="enhance: loading audio")
            payload = task_manager.get_payload(task_id)
            audio_bytes = payload.get("audio_bytes") if payload else None
            audio, sr, duration = load_audio(audio_bytes)
            task_manager.update_task(task_id, progress=0.10, message=f"enhance: audio loaded sr={sr} dur={duration:.2f}s")

            if not self.enhance_model:
                task_manager.update_task(task_id, stage_name="enhanced", stage_status="error", stage_result={"error":"enhance model not configured"}, message="enhance: model missing")
                return

            out = self.enhance_model.generate(input=audio, sample_rate=sr)
            enhanced_audio = out.get("audio", audio) if isinstance(out, dict) else audio
            task_manager.update_task(task_id, stage_name="enhanced", stage_status="done", stage_result={"duration": duration})

            # 覆盖 payload 中的音频以供后续阶段使用
            new_payload = {"audio_bytes": audio_bytes, "audio": enhanced_audio, "sr": sr, "duration": duration}
            task_manager.update_task(task_id, payload=new_payload, message="enhance: done")
        except Exception as e:
            task_manager.update_task(task_id, stage_name="enhanced", stage_status="error", stage_result={"error": str(e)}, message="enhance: error")

    async def run_enhanced_task(self, task_id, task_manager):
        await asyncio.to_thread(self._enhanced_sync, task_id, task_manager)

    # ===== 单阶段：官方 VAD =====
    def _vad_sync(self, task_id, task_manager):
        try:
            task_manager.update_task(task_id, status="running", progress=0.05, message="vad: loading audio")
            payload = task_manager.get_payload(task_id)
            audio_bytes = payload.get("audio_bytes") if payload else None
            audio, sr, duration = load_audio(audio_bytes)
            task_manager.update_task(task_id, progress=0.10, message=f"vad: audio loaded sr={sr} dur={duration:.2f}s")

            if not self.vad_model:
                task_manager.update_task(task_id, stage_name="diarization", stage_status="error", stage_result={"error":"vad model not configured"}, message="vad: model missing")
                return

            out = self.vad_model.generate(input=audio, sample_rate=sr)
            segments = out.get("segments", []) if isinstance(out, dict) else []
            if not segments:
                segments = [{"start": 0.0, "end": duration}]
            task_manager.update_task(task_id, stage_name="diarization", stage_status="done", stage_result={"segments": segments}, message=f"vad: {len(segments)} segments")
        except Exception as e:
            task_manager.update_task(task_id, stage_name="diarization", stage_status="error", stage_result={"error": str(e)}, message="vad: error")

    async def run_vad_task(self, task_id, task_manager):
        await asyncio.to_thread(self._vad_sync, task_id, task_manager)

    # ===== 单阶段：使用 SenseSmallVoice 识别 =====
    def _transformer_sync(self, task_id, task_manager):
        try:
            task_manager.update_task(task_id, status="running", progress=0.05, message="transformer: loading audio")
            payload = task_manager.get_payload(task_id)
            audio_bytes = payload.get("audio_bytes") if payload else None
            audio, sr, duration = load_audio(audio_bytes)
            task_manager.update_task(task_id, progress=0.10, message=f"transformer: audio loaded sr={sr} dur={duration:.2f}s")

            model = self.sense_model if self.sense_model is not None else self.asr_model
            if model is None:
                task_manager.update_task(task_id, stage_name="transformer", stage_status="error", stage_result={"error":"SenseSmallVoice not configured"}, message="transformer: model missing")
                return

            out = model.generate(input=audio, sample_rate=sr)
            text = out[0]["text"] if isinstance(out, list) and len(out) else ""
            task_manager.update_task(task_id, stage_name="transformer", stage_status="done", stage_result={"text": text, "duration": duration}, message="transformer: done")
        except Exception as e:
            task_manager.update_task(task_id, stage_name="transformer", stage_status="error", stage_result={"error": str(e)}, message="transformer: error")

    async def run_transformer_task(self, task_id, task_manager):
        await asyncio.to_thread(self._transformer_sync, task_id, task_manager)

    # ===== 全流程：增强 ->（必要时）分离/分段 -> ASR =====
    def _run_full_sync(self, task_id, audio_bytes, task_manager):
        try:
            task_manager.update_task(task_id, status="running", progress=0.05, message="loading audio")
            audio, sr, duration = load_audio(audio_bytes)
            task_manager.update_task(task_id, payload={"audio_bytes": audio_bytes, "audio": audio, "sr": sr, "duration": duration}, progress=0.10, message=f"audio loaded: sr={sr}, dur={duration:.2f}s")

            # 1) 增强
            if self.enhance_model:
                try:
                    enh_out = self.enhance_model.generate(input=audio, sample_rate=sr)
                    audio = enh_out.get("audio", audio) if isinstance(enh_out, dict) else audio
                    task_manager.update_task(task_id, stage_name="enhanced", stage_status="done", stage_result={"duration": duration}, message="enhance: done")
                except Exception as e:
                    task_manager.update_task(task_id, stage_name="enhanced", stage_status="error", stage_result={"error": str(e)}, message="enhance: error")

            # 2) 分段（官方 VAD 优先）
            segments = []
            if self.vad_model:
                try:
                    out = self.vad_model.generate(input=audio, sample_rate=sr)
                    segments = out.get("segments", []) if isinstance(out, dict) else []
                    if not segments:
                        segments = [{"start": 0.0, "end": duration}]
                    task_manager.update_task(task_id, stage_name="diarization", stage_status="done", stage_result={"segments": segments}, message=f"vad: {len(segments)} segments")
                except Exception as e:
                    segments = [{"start": 0.0, "end": duration}]
                    task_manager.update_task(task_id, stage_name="diarization", stage_status="error", stage_result={"error": str(e)}, message="vad: error; using full audio")
            else:
                segments = [{"start": 0.0, "end": duration}]
                task_manager.update_task(task_id, message="vad: not configured; using full audio")

            # 3) 若具备真实分离/分离模型则使用其输出（覆盖 segments）
            if self.use_diarization and self.spk_model is not None:
                try:
                    diar_result = self.spk_model.generate(input=audio, sample_rate=sr)
                    spk_segments = diar_result.get("segments", []) if isinstance(diar_result, dict) else []
                    if spk_segments:
                        segments = spk_segments
                        task_manager.update_task(task_id, message=f"diarization: {len(segments)} segments with speakers")
                except Exception as e:
                    task_manager.update_task(task_id, message=f"diarization: failed {e}")

            # 4) 分段识别
            results = []
            total = len(segments)
            for i, seg in enumerate(segments):
                seg_audio = audio[int(seg["start"]*sr): int(seg["end"]*sr)]
                asr_out = self.asr_model.generate(input=seg_audio, sample_rate=sr)
                text = asr_out[0]["text"] if isinstance(asr_out, list) and len(asr_out) else ""
                results.append({"start": seg["start"], "end": seg["end"], "text": text})
                task_manager.update_task(task_id, progress=0.20 + 0.70*(i+1)/max(total,1), message=f"asr: {i+1}/{total}")

            # 5) 合并文本与时间戳
            final_text = combine_segments(results)
            task_manager.update_task(task_id, status="done", result={"text": final_text, "segments": results, "duration": duration}, message="done")
        except Exception as e:
            task_manager.update_task(task_id, status="error", error=str(e), message="pipeline error")

    async def run_full_task(self, task_id, audio_bytes, task_manager):
        await asyncio.to_thread(self._run_full_sync, task_id, audio_bytes, task_manager)

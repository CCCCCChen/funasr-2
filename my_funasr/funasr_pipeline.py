import io, librosa
from funasr import AutoModel
from my_funasr.audio_preprocess import load_audio
from my_funasr.text_postprocess import combine_segments

class FunASRPipeline:
    def __init__(self, asr_model_dir, punc_model_dir, spk_model_dir, device="cpu"):
        print("🚀 Loading ASR, punctuation, and speaker models...")
        self.asr_model = AutoModel(model=asr_model_dir, device=device)
        self.punc_model = AutoModel(model=punc_model_dir, device=device)
        self.spk_model = AutoModel(model=spk_model_dir, device=device)
        print("✅ Models loaded.")

    async def run_task(self, task_id, audio_bytes, task_manager):
        try:
            task_manager.update_task(task_id, status="running")

            # 加载音频
            audio, sr, duration = load_audio(audio_bytes)

            # 说话人分离（返回 segments 带说话人标签）
            diar_result = self.spk_model.generate(input=audio, sample_rate=sr)
            segments = diar_result["segments"]  # [{'start':, 'end':, 'spk':}, ...]

            # 分段识别
            results = []
            for seg in segments:
                seg_audio = audio[int(seg["start"]*sr): int(seg["end"]*sr)]
                asr_out = self.asr_model.generate(input=seg_audio)
                text = asr_out[0]["text"]

                # 标点恢复
                punc_out = self.punc_model.generate(input=text)
                final_text = punc_out[0]["text"]

                results.append({
                    "speaker": seg["spk"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": final_text
                })

            combined_text = combine_segments(results)

            task_manager.update_task(task_id, status="done", result={
                "text": combined_text,
                "segments": results,
                "duration": duration
            })
        except Exception as e:
            task_manager.update_task(task_id, status="error", result={"error": str(e)})

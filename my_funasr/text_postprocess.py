def combine(results):
    full_text = ""
    segments = []
    t = 0.0
    for r in results:
        seg_text = r["output"]["text"]
        dur = r["duration"]
        segments.append({
            "start": t,
            "end": t + dur,
            "text": seg_text
        })
        full_text += seg_text
        t += dur
    return full_text.strip(), segments

# 新增：兼容 FunASR 管线的分段合并
# 输入: results = [{"speaker": str, "start": float, "end": float, "text": str}, ...]
# 输出: 合并后的整段文本字符串
def combine_segments(results):
    texts = []
    for r in results:
        # 可选：在文本前标注说话人，例如 f"[{r['speaker']}] {r['text']}"
        # 当前为简洁输出，仅拼接文字
        texts.append(r.get("text", ""))
    return "".join(texts).strip()

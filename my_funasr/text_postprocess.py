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

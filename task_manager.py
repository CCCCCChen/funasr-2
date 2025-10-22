import threading

class TaskManager:
    def __init__(self):
        self.tasks = {}
        self.lock = threading.Lock()

    def create_task(self, task_id, status="pending", payload=None):
        with self.lock:
            self.tasks[task_id] = {
                "status": status,
                "result": None,
                "progress": 0.0,
                "error": None,
                "message": None,
                "payload": payload,  # e.g., {"audio_bytes": ..., "audio": ndarray, "sr": 16000, "duration": 12.3}
                "stages": {},        # per-stage results/status, e.g., {"enhanced": {...}, "vad": {...}}
            }

    def update_task(self, task_id, status=None, result=None, progress=None, error=None, message=None, payload=None, stage_name=None, stage_result=None, stage_status=None):
        with self.lock:
            if task_id not in self.tasks:
                return
            if status is not None:
                self.tasks[task_id]["status"] = status
            if result is not None:
                self.tasks[task_id]["result"] = result
            if progress is not None:
                # clamp to [0.0, 1.0]
                try:
                    p = float(progress)
                except Exception:
                    p = self.tasks[task_id].get("progress", 0.0)
                self.tasks[task_id]["progress"] = max(0.0, min(1.0, p))
            if error is not None:
                self.tasks[task_id]["error"] = error
            if message is not None:
                self.tasks[task_id]["message"] = message
            if payload is not None:
                self.tasks[task_id]["payload"] = payload
            # per-stage structured updates
            if stage_name is not None:
                stages = self.tasks[task_id].setdefault("stages", {})
                stage_entry = stages.setdefault(stage_name, {})
                if stage_result is not None:
                    stage_entry["result"] = stage_result
                if stage_status is not None:
                    stage_entry["status"] = stage_status

    def get_task(self, task_id):
        with self.lock:
            return self.tasks.get(task_id)

    def set_payload(self, task_id, payload):
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id]["payload"] = payload

    def get_payload(self, task_id):
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            return task.get("payload")

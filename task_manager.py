import threading

class TaskManager:
    def __init__(self):
        self.tasks = {}
        self.lock = threading.Lock()

    def create_task(self, task_id, status="pending"):
        with self.lock:
            self.tasks[task_id] = {
                "status": status,
                "result": None,
                "progress": 0.0,
                "error": None,
                "message": None,
            }

    def update_task(self, task_id, status=None, result=None, progress=None, error=None, message=None):
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

    def get_task(self, task_id):
        with self.lock:
            return self.tasks.get(task_id)

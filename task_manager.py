import threading

class TaskManager:
    def __init__(self):
        self.tasks = {}
        self.lock = threading.Lock()

    def create_task(self, task_id, status="pending"):
        with self.lock:
            self.tasks[task_id] = {"status": status, "result": None}

    def update_task(self, task_id, status=None, result=None):
        with self.lock:
            if task_id not in self.tasks:
                return
            if status:
                self.tasks[task_id]["status"] = status
            if result is not None:
                self.tasks[task_id]["result"] = result

    def get_task(self, task_id):
        with self.lock:
            return self.tasks.get(task_id)

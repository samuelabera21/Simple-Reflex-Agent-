from datetime import datetime
from pathlib import Path


class ActionLogger:
    def __init__(self, log_file="actions.log"):
        self.log_path = Path(log_file)
        if not self.log_path.exists():
            self.log_path.touch()

    def log_action(self, action):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{timestamp} - {action}\n")

    def clear_logs(self):
        self.log_path.write_text("", encoding="utf-8")

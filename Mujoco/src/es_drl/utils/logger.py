# src/es_drl/utils/logger.py
"""
Logger utility for tracking training progress.
This class provides functionality for logging training metrics to CSV files,
with support for appending to existing logs and automatic header management.
"""

import os


class Logger:
    def __init__(self, log_dir: str, filename: str = "progress.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, filename)

        # If file exists and is non-empty, open in append and read header
        if os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0:
            self.file = open(self.filepath, "a")
            with open(self.filepath, "r") as f:
                header = f.readline().strip().split(",")[1:]
            self.keys = header
            self.header_written = True
        else:
            # Create fresh file (write mode)
            self.file = open(self.filepath, "w")
            self.keys = []
            self.header_written = False

    def log(self, step: int, data: dict):
        # First time: write header
        if not self.header_written:
            self.keys = list(data.keys())
            self.file.write("step," + ",".join(self.keys) + "\n")
            self.header_written = True

        # Write data line
        values = [str(data[k]) for k in self.keys]
        self.file.write(f"{step}," + ",".join(values) + "\n")
        self.file.flush()

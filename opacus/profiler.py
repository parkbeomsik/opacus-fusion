from opacus import config

import torch
import time
from collections import defaultdict
import gc

import pandas as pd

def get_memory_usage():
    # Print all tensors
    gc.collect()
    memory_usage = {}
    for obj in gc.get_objects():
        try:
            if (torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data))) \
                and obj.device == "cuda:0":
                if type(obj) == torch.Tensor:
                    pass
        except:
            pass

class Profiler():
    def __init__(self):
        self.time_records = defaultdict(int)
        self.memory_records = defaultdict(int)
        self.start_step_time = time.time()
        self.start_interval_time = time.time()
        self.ignore_time = 0
        self.time_keys = ["Total", "Data loading", "Forward", "Backward activation", "Backward weight", "Clip/reduce", "Add noise", "Update"]
        self.memory_keys = ["Peak memory usage"]
        self.peak_memory_usage = 0
        self.step_count = 0

    def init_step(self):
        torch.cuda.synchronize()
        self.start_step_time = time.time()
        self.start_interval_time = time.time()
        self.ignore_time = 0

    def end_step(self):
        torch.cuda.synchronize()
        self.time_records["Total"] += (time.time() - self.start_step_time) * 1000 - self.ignore_time # ms
        self.peak_memory_usage = torch.cuda.max_memory_allocated()
        self.step_count += 1
        self.init_step()

    def reset_time(self):
        torch.cuda.synchronize()
        self.ignore_time += (time.time() - self.start_interval_time) * 1000 # ms
        self.start_interval_time = time.time()

    def record_time(self, type=""):
        torch.cuda.synchronize()
        self.time_records[type] += (time.time() - self.start_interval_time) * 1000 # ms
        self.start_interval_time = time.time()

    def add_time_explicit(self, type="", time_elapsed=0):
        self.time_records[type] += time_elapsed
        torch.cuda.synchronize()
        self.start_interval_time = time.time()

    def record_memory(self, type=""):
        self.memory_records["Peak memory usage"] = torch.cuda.max_memory_allocated() # Bytes

    def record(self, type=""):
        if config.profile_time:
            self.record_time(type)
        if config.profile_memory:
            self.record_memory(type)

    def time_as_df(self, index):
        time_data = [[self.time_records[key] / self.step_count for key in self.time_keys] + [self.peak_memory_usage]]
        time_df = pd.DataFrame(time_data, columns=self.time_keys + ["Peak memory usage"], index=index)
        return time_df

    def memory_as_df(self, index):
        memory_data = [[self.memory_records[key] for key in self.memory_keys]]
        memory_df = pd.DataFrame(memory_data, columns=self.memory_keys, index=index)
        return memory_df

profiler = Profiler()

total_ignored_time = []
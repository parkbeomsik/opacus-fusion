from email.policy import default
from opacus import config

import torch
import time
from collections import defaultdict
import gc
import traceback

import pandas as pd

from opacus.custom_tensor import PerBatchGrads, PerSampleGrads, GradOutputs

def get_memory_usage(input_activation=False):
    # Print all tensors
    gc.collect()
    memory_usage = defaultdict(int)
    tracked_ptr_set = set()
    for obj in gc.get_objects():
        try:
            if (torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data))) \
                and obj.get_device() == 0:
                size_in_bytes = obj.numel() * obj.element_size()
                if type(obj) == torch.Tensor:
                    if input_activation and (obj.data_ptr() not in tracked_ptr_set):
                        memory_usage["Input activations"] += size_in_bytes
                elif type(obj) == torch.nn.Parameter:
                    memory_usage["Weights"] += size_in_bytes
                elif type(obj) == PerSampleGrads:
                    memory_usage["Per-example weight gradients"] += size_in_bytes
                elif type(obj) == PerBatchGrads:
                    memory_usage["Per-batch weight gradients"] += size_in_bytes
                elif type(obj) == GradOutputs:
                    memory_usage["Activation gradients"] += size_in_bytes
                tracked_ptr_set.add(obj.data_ptr())
        except:
            pass

    # if memory_usage["Input activations"] > 18000000000:
    #     traceback.print_stack()
    #     for obj in gc.get_objects():
    #         try:
    #             if (torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data))) \
    #                 and obj.get_device() == 0:
    #                 size_in_bytes = obj.numel() * obj.element_size()
    #                 if type(obj) == torch.Tensor:
    #                     print(obj.shape)
    #         except:
    #             pass

    #     exit(0)

    return memory_usage

class Profiler():
    def __init__(self):
        self.time_records = defaultdict(int)
        self.memory_records = defaultdict(int)
        self.start_step_time = time.time()
        self.start_interval_time = time.time()
        self.ignore_time = 0
        self.time_keys = ["Total", "Data loading", "Forward", "Backward activation", "Backward weight", "Clip/reduce", "Add noise", "Update"]
        self.memory_keys = ["Peak memory usage", "Weights", "Input activations", "Activation gradients", "Per-batch weight gradients", "Per-example weight gradients"]
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

    def record_memory(self, type="", input_activation=False):
        if config.profile_memory:
            torch.cuda.synchronize()
            self.memory_records["Peak memory usage"] = torch.cuda.max_memory_allocated() # Bytes
            memory_usage = get_memory_usage(input_activation)
            for key in memory_usage:
                if not input_activation and key == "Input activations":
                    continue
                self.memory_records[key] = max(memory_usage[key], self.memory_records[key])
            torch.cuda.synchronize()


    def record(self, type="", input_activation=False):
        if config.profile_time:
            self.record_time(type)
        if config.profile_memory:
            self.record_memory(type, input_activation=input_activation)

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
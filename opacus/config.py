from enum import Enum

MODE_NAIVE = 1
MODE_REWEIGHT = 2
MODE_ELEGANT = 3

# reweight = False
layer_type = "Conv"
model_type = "cnn"
architecture = "gnmt"
# example_wise = True
dpsgd_mode = MODE_ELEGANT
selective_reweight = True
quantization = False
grad_sample_mode = "hooks"

batch_size = 0

verbose = True

precomputed_grads = None


# Profiler
profile_time = True
profile_memory = False
profile_throughput = True
profile_value = False

grad_save_path = None

warm_up_steps = 0
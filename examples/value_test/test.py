import subprocess
import os

from test_equal import is_equal

def execute_command(cmd):
    ret = subprocess.run(cmd, shell=True, capture_output=True)
    if ret.returncode != 0:
        print(f"Error")
        print(cmd)
        print(ret.stderr.decode())
        assert 0

if not os.path.exists("value_data"):
    os.mkdir("value_data")
if not os.path.exists("grad_sample"):
    os.mkdir("grad_sample")

case_list = ["cnn resnet18", "cnn resnet50", "cnn resnet152"]

for case in case_list:
    model_type, architecture = case.split(" ")
    print(f"============================================")
    print(f"Test {model_type} ({architecture})...")

    # Create fixed weight and input
    execute_command(f"python benchmark.py --input_size 32 --architecture {architecture} --model_type {model_type} --dpsgd_mode naive --batch_size 4 --model_save_path value_test/value_data/{architecture}_weight.pt --input_save_path value_test/value_data/{architecture}_input.pt -c 0.1 --sigma 0.0")

    # Create reference grads (naive DPSGD)
    print("Execute naive...")
    execute_command(f"python benchmark.py --input_size 32 --architecture {architecture} --model_type {model_type} --dpsgd_mode naive --batch_size 16 --model_load_path value_test/value_data/{architecture}_weight.pt --input_load_path value_test/value_data/{architecture}_input.pt --grad_save_path value_test/value_data/{architecture}_grad_naive.pt --profile_value -c 0.1 --sigma 0.0")

    # Test grads of DPSGD(R)
    print("Execute reweight...")
    execute_command(f"python benchmark.py --input_size 32 --architecture {architecture} --model_type {model_type} --dpsgd_mode reweight --batch_size 16 --model_load_path value_test/value_data/{architecture}_weight.pt --input_load_path value_test/value_data/{architecture}_input.pt --grad_save_path value_test/value_data/{architecture}_grad_reweight.pt --profile_value -c 0.1 --sigma 0.0")

    reweight_ret = is_equal(f"value_test/value_data/{architecture}_grad_naive.pt",
                            f"value_test/value_data/{architecture}_grad_reweight.pt",
                            verbose=False)

    if reweight_ret:
        print("Results of reweight is correct!")
    else:
        print("Results of reweight is wrong!")
        is_equal(f"value_test/value_data/{architecture}_grad_naive.pt",
                 f"value_test/value_data/{architecture}_grad_reweight.pt",
                 verbose=True)

    # Test grads of elegant
    print("Execute elegant ...")
    execute_command(f"python benchmark.py --steps 1 --input_size 32 --architecture {architecture} --model_type {model_type} --dpsgd_mode elegant --batch_size 16 --model_load_path value_test/value_data/{architecture}_weight.pt --input_load_path value_test/value_data/{architecture}_input.pt --grad_save_path value_test/value_data/{architecture}_grad_elegant.pt --profile_value -c 0.1 --sigma 0.0")

    elegant_ret = is_equal(f"value_test/value_data/{architecture}_grad_naive.pt",
                           f"value_test/value_data/{architecture}_grad_elegant.pt",
                           verbose=False)

    if elegant_ret:
        print("Results of elegant is correct!")
    else:
        print("Results of elegant is wrong!")
        is_equal(f"value_test/value_data/{architecture}_grad_naive.pt",
                 f"value_test/value_data/{architecture}_grad_elegant.pt",
                 verbose=True)
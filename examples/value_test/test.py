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

if not os.path.exists("value_test/value_data"):
    os.mkdir("value_test/value_data")
if not os.path.exists("value_test/grad_sample"):
    os.mkdir("value_test/grad_sample")

def test(model_type, architecture, dpsgd_mode, grad_sample_mode):
    print(f"Execute {dpsgd_mode} ({grad_sample_mode})...")
    execute_command(f"python benchmark.py --input_size 32 --architecture {architecture} --model_type {model_type} --dpsgd_mode {dpsgd_mode} --batch_size 2 --model_load_path value_test/value_data/{architecture}_weight.pt --input_load_path value_test/value_data/{architecture}_input.pt --grad_save_path value_test/value_data/{architecture}_grad_{dpsgd_mode}_{grad_sample_mode}.pt --profile_value -c 1 --sigma 0.0 --grad_sample_mode {grad_sample_mode}")

    ret = is_equal(f"value_test/value_data/{architecture}_grad_ref.pt",
                   f"value_test/value_data/{architecture}_grad_{dpsgd_mode}_{grad_sample_mode}.pt",
                    verbose=False)

    if ret:
        print(f"Results of {dpsgd_mode} ({grad_sample_mode}) is correct!")
    else:
        print(f"Results of {dpsgd_mode} ({grad_sample_mode}) is wrong!")
        is_equal(f"value_test/value_data/{architecture}_grad_ref.pt",
                 f"value_test/value_data/{architecture}_grad_{dpsgd_mode}_{grad_sample_mode}.pt",
                 verbose=True)


cnn_case_list = ["cnn resnet18", "cnn resnet50", "cnn resnet152"]
transformer_case_list = ["transformer bert-base"]#, "transformer bert-large"]
rnn_case_list = ["rnn gnmt"]

case_list = rnn_case_list 

for case in case_list:
    model_type, architecture = case.split(" ")
    print(f"============================================")
    print(f"Test {model_type} ({architecture})...")

    # Create fixed weight and input
    execute_command(f"python benchmark.py --input_size 32 --architecture {architecture} --model_type {model_type} --dpsgd_mode naive --batch_size 2 --model_save_path value_test/value_data/{architecture}_weight.pt --input_save_path value_test/value_data/{architecture}_input.pt -c 1 --sigma 0.0 --warm_up_steps 0 --steps 1")

    # Create reference grads (naive DPSGD)
    print("Execute naive...")
    execute_command(f"python benchmark.py --input_size 32 --architecture {architecture} --model_type {model_type} --dpsgd_mode naive --batch_size 2 --model_load_path value_test/value_data/{architecture}_weight.pt --input_load_path value_test/value_data/{architecture}_input.pt --grad_save_path value_test/value_data/{architecture}_grad_ref.pt --profile_value -c 1 --sigma 0.0 --grad_sample_mode hooks")

    execute_command(f"python benchmark.py --input_size 32 --architecture {architecture} --model_type {model_type} --disable-dp --batch_size 2 --model_load_path value_test/value_data/{architecture}_weight.pt --input_load_path value_test/value_data/{architecture}_input.pt --grad_save_path value_test/value_data/{architecture}_grad_no_dp.pt --profile_value -c 1 --sigma 0.0 --grad_sample_mode hooks")

    # Test grads of DPSGD with ExpandedWeights
    # test(model_type, architecture, "naive", "ew")

    # Test grads of DPSGD(R)
    # test(model_type, architecture, "reweight", "hooks")

    # Test grads of DPSGD(R) with ExpandedWeights
    # test(model_type, architecture, "reweight", "ew")

    # Test grads of Elegant
    test(model_type, architecture, "elegant", "hooks")

import subprocess

model_cand = ["cnn resnet18", "cnn resnet50", "cnn resnet152"]

for architecture, model in model_cand:
    

ret = subprocess.run("nice -n 10 /home/beomsik/miniconda3/envs/fusion/bin/python benchmark.py --steps 1 --input_size 32 --architecture resnet152 --model_type cnn --dpsgd_mode naive --batch_size 4 --model_save_path value_test/value_data/resnet152_weight.pt --input_save_path value_test/value_data/resnet152_input.pt -c 0.1 --sigma 0.0", shell=True, capture_output=True)

ret = subprocess.run("nice -n 10 /home/beomsik/miniconda3/envs/fusion/bin/python benchmark.py --steps 1 --input_size 32 --architecture resnet152 --model_type cnn --dpsgd_mode naive --batch_size 16 --model_load_path value_test/value_data/resnet152_weight.pt --input_load_path value_test/value_data/resnet152_input.pt --grad_save_path value_test/value_data/resnet152_grad_naive.pt --profile_value -c 0.1 --sigma 0.0", shell=True, capture_output=True)
if ret.returncode != 0:
    print("[Failed] Full command : nice -n 10 /home/beomsik/miniconda3/envs/fusion/bin/python benchmark.py --steps 1 --input_size 32 --architecture sample_conv_net --model_type cnn --dpsgd_mode naive --batch_size 16 --model_load_path value_test/value_data/sample_conv_net_weight.pt --input_load_path value_test/value_data/sample_conv_net_input.pt --grad_save_path value_test/value_data/sample_conv_net_grad_naive.pt --profile_value -c 0.1")

subprocess.run("nice -n 10 /home/beomsik/miniconda3/envs/fusion/bin/python benchmark.py --steps 1 --input_size 32 --architecture sample_conv_net --model_type cnn --dpsgd_mode naive --batch_size 16 --model_load_path value_test/value_data/sample_conv_net_weight.pt --input_load_path value_test/value_data/sample_conv_net_input.pt --grad_save_path value_test/value_data/sample_conv_net_grad_naive.pt --profile_value -c 0.1", shell=True, capture_output=True)
subprocess.run("nice -n 10 /home/beomsik/miniconda3/envs/fusion/bin/python benchmark.py --steps 1 --input_size 32 --architecture sample_conv_net --model_type cnn --dpsgd_mode reweight --batch_size 16 --model_load_path value_test/value_data/sample_conv_net_weight.pt --input_load_path value_test/value_data/sample_conv_net_input.pt --grad_save_path value_test/value_data/sample_conv_net_grad_reweight.pt --profile_value -c 0.1", shell=True, capture_output=True)
subprocess.run("nice -n 10 /home/beomsik/miniconda3/envs/fusion/bin/python benchmark.py --steps 1 --input_size 32 --architecture sample_conv_net --model_type cnn --dpsgd_mode elegant --batch_size 16 --model_load_path value_test/value_data/sample_conv_net_weight.pt --input_load_path value_test/value_data/sample_conv_net_input.pt --grad_save_path value_test/value_data/sample_conv_net_grad_elegant.pt --profile_value -c 0.1", shell=True, capture_output=True)

subprocess.run("nice -n 10 /home/beomsik/miniconda3/envs/fusion/bin/python benchmark.py --steps 1 --input_size 32 --architecture resnet18 --model_type cnn --dpsgd_mode naive --batch_size 16 --model_load_path value_test/value_data/resnet18_weight.pt --input_load_path value_test/value_data/resnet18_input.pt --grad_save_path value_test/value_data/resnet18_grad_naive.pt --profile_value -c 0.1 --sigma 0.0", shell=True, capture_output=True)
import shutil
import subprocess
import random
import json
import datetime
import os
import argparse
import glob

parser = argparse.ArgumentParser(description='Accuracy test on NLP fine-tuning with DP-SGD and int8 quantization')
parser.add_argument('--n_trials', metavar='N', type=int, default=10,
                    help='number of trials (to be averaged)')
parser.add_argument('--task', type=str, default="sst-2", choices=['sst-2', 'qnli', 'qqp', 'mnli'],
                    help='fine-tuning task name (default="sst-2")')
parser.add_argument('--quant_mode', type=str, default="no", choices=['no', 'int8'],
                    help='quant mode (default="no")')
parser.add_argument('--model', type=str, default="roberta-base", choices=["roberta-base", "roberta-large", "bert-base", "bert-large"],
                    help='model name (default="roberta-base")')
parser.add_argument('--per_device_train_batch_size', metavar='N', type=int, default=20,
                    help=' Per device train batch size (non virtual)')

if __name__ == "__main__":
    args = parser.parse_args()
    
    num_trials = args.n_trials
    task_name = args.task
    total_results = []
    total_seeds = []
    quant_mode = args.quant_mode
    model_name = args.model
    per_device_train_batch_size = args.per_device_train_batch_size

    random.seed(100)
    seeds = [random.randint(0, 10000) for _ in range(num_trials)]

    summary_f = open(f"summary/{task_name}_{model_name}_{quant_mode}_per-batch_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}_result.csv", "w")
    summary_f.write(f"Configurations,{task_name},{model_name},{quant_mode},eps=7.97,{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}\n")
    if task_name == "mnli":
        summary_f.write(f"seed,eval_mnli/acc,eval_mnli-mm/acc\n")
    else:
        summary_f.write(f"seed,eval_acc\n")

    # Check previous experiments
    line_count_list = []
    for file_name in glob.glob(f"summary/{task_name}_{model_name}_{quant_mode}_per-batch*"):
        with open(file_name, "r") as f:
            line_count_list.append((file_name, len(f.readlines())))
    max_line_prev_exp_file_name = max(line_count_list, key=lambda x: x[1])[0]
    num_completed_exps = 0
    with open(max_line_prev_exp_file_name, "r") as f:
        line = f.readline()
        while line:
            if "seed" != line[:4] and "Configurations" != line[:14]:
                summary_f.write(line)
                num_completed_exps += 1
            line = f.readline()
    summary_f.flush()

    for i in range(num_completed_exps, num_trials):
        print(f"=============== Trial {i} ===============")
        print()
        seed = seeds[i]
        print(task_name, model_name, quant_mode)
        
        cmd_list = ["taskset", "-c", "21-40", "python", f"-m", f"classification.run_wrapper", f"--output_dir", f"{task_name}_{model_name}_{quant_mode}", f"--task_name", f"{task_name}", f"--few_shot_type", f"prompt", f"--clipping_mode", f"default ", f"--seed", f"{seed}", f"--quant_mode", f"{quant_mode}", f"--model_name_or_path", f"{model_name}", f"--per_device_train_batch_size", f"{per_device_train_batch_size}"]
        subprocess.run(cmd_list, stdout=None, stderr=None)

        with open(f"{task_name}_{model_name}_{quant_mode}/final_results.json", "r") as f:
            cur_results = json.load(f)
            if task_name == "mnli":
                total_results += [str(cur_results[task_name]["eval_mnli/acc"]) + "," + str(cur_results["mnli-mm"]["eval_mnli-mm/acc"])]
            else:
                total_results += [cur_results[task_name]["eval_acc"]]
            total_seeds += [seed]
            print(f"Seed : {seed}\n")
            if task_name == "mnli":
                print(f"Eval acc : {cur_results['mnli']['eval_mnli/acc']},{cur_results['mnli-mm']['eval_mnli-mm/acc']}")
            else:
                print(f"Eval acc : {cur_results[task_name]['eval_acc']}")
            print()

            if task_name == "mnli":
                summary_f.write(f"{seed},{cur_results['mnli']['eval_mnli/acc']},{cur_results['mnli-mm']['eval_mnli-mm/acc']}\n")
            else:
                summary_f.write(f"{seed},{cur_results[task_name]['eval_acc']}\n")
            summary_f.flush()

    print("================================================================")
    print("==================  All experiments finished  ==================")
    print("================================================================")
    print(total_results)

    shutil.rmtree(f"{task_name}_{model_name}_{quant_mode}")

    summary_f.close()

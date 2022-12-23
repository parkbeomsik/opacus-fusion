import os
import argparse
from tabulate import tabulate

parser = argparse.ArgumentParser(description='Get statistics of summary')
parser.add_argument('--dir', type=str, default="examples/summary",
                    help='Summary directory')

def get_statistics(files):
    statistics = {}

    for file in files:
        model = file.split('_')[1]
        task = file.split('_')[0]
        quant = file.split('_')[2]
        timestamp = file.split('_', 3)[3]

        if (model, task, quant) in statistics.keys():
            if statistics[(model, task, quant)]['timestamp'] < timestamp:
                del statistics[(model, task, quant)]
            else:
                continue

        results = []
        results_mm = []
        with open(os.path.join(dir, file), 'r') as f:
            lines = f.readlines()

            for line in lines[2:]:
                line = line.strip()
                if line == "":
                    break
                results += [float(line.split(',')[1]) * 100]
                if task=="mnli":
                    results_mm += [float(line.split(',')[2]) * 100]
        
        if results == []:
            continue
        
        statistics[(model, task, quant)] = {'timestamp': timestamp,
                                            'results': results,
                                            'mean': sum(results)/len(results),
                                            'max': max(results),
                                            'min': min(results)}

        if task == "mnli":
            statistics[(model, "mnli-mm", quant)] = {'timestamp': timestamp,
                                                'results': results_mm,
                                                'mean': sum(results_mm)/len(results_mm),
                                                'max': max(results_mm),
                                                'min': min(results_mm)}

    return statistics

def sort_score(key):
    model_list = ["roberta-base", "roberta-large"]
    task_list = ["sst-2", "qnli", "qqp", "mnli", "mnli-mm"]
    quant_list = ["no", "int8"]

    return model_list.index(key[0])*100 + task_list.index(key[1])*10 + quant_list.index(key[2])

if __name__ == "__main__":
    args = parser.parse_args()

    dir = args.dir

    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    statistics = get_statistics(files)

    # print statistics

    keys = sorted(statistics.keys(), key=sort_score)

    table = []
    for key in keys:
        table += [[*key, statistics[key]['mean'], statistics[key]['max'], statistics[key]['min']]]
    

    print(tabulate(table,
                   headers=["Model", "Task", "Quant", "Mean", "Max", "Min"],
                   tablefmt="grid",
                   floatfmt=".2f"))
# Set environment
CPU_AFFINITY=0-2
LOG_FILE_NAME=results/result.csv

# Batch size configuration
declare -A BATCH_DICT=( ["deepspeech 128"]="16" 
                        
                        ["gnmt 128"]="16" 
                        ["gnmt 128"]="16" 
                        ["gnmt 128"]="16" 
                        ["bert-base"]=""
                        ["bert-large"]="")
# echo $CPU_AFFINITY

nice -n 10 taskset -c $CPU_AFFINITY python benchmark.py --steps 10 --input_size 128 --architecture gnmt --model_type rnn --dpsgd_mode elegant --batch_size 4 --profile_time --log_file $LOG_FILE_NAME
nice -n 10 taskset -c $CPU_AFFINITY python benchmark.py --steps 10 --input_size 128 --architecture gnmt --model_type rnn --dpsgd_mode elegant --batch_size 4 --profile_time --log_file $LOG_FILE_NAME
nice -n 10 taskset -c $CPU_AFFINITY python benchmark.py --steps 10 --input_size 128 --architecture gnmt --model_type rnn --dpsgd_mode elegant --batch_size 4 --profile_time --log_file $LOG_FILE_NAME
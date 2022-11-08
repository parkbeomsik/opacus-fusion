## Prerequisite
CUDA Toolkit, CUDNN, CUBLAS should be installed.

## Installation

0. Set envrionment variables
```bash
export ENV_NAME={ENV_NAME}
export OPACUS_FUSION_PATH={OPACUS_FUSION_PATH} # Absolute path
export CUTLASS_PATH={CUTLASS_PATH} # Absolute path
```
1. Create conda envrionment
```bash
conda create -n $ENV_NAME python=3.9
conda activate $ENV_NAME
```
2. Install torch from https://pytorch.org/get-started/locally/
3. Download opacus-fusion from https://github.com/parkbeomsik/opacus-fusion
```bash
git clone https://github.com/parkbeomsik/opacus-fusion.git $OPACUS_FUSION_PATH
```
4. Download cutlass from https://github.com/parkbeomsik/cutlass
```bash
git clone https://github.com/parkbeomsik/cutlass.git $CUTLASS_PATH
```
5. Install cutlass_wgrad_grouped (It will create `lib` and `include` in `build` directory)
```bash
cd $OPACUS_FUSION_PATH
cd cutlass_wgrad_grouped
mkdir build && cd build
cmake .. -DCUTLASS_PATH=$CUTLASS_PATH
make install
```
6. Install grad_example_module
```bash
cd $OPACUS_FUSION_PATH
cd grad_example_module
python setup.py install
```
7. Install custom_rnn
```bash
cd $OPACUS_FUSION_PATH
cd custom_rnn
python setup.py install
```
8. Install opacus-fusion
```bash
cd $OPACUS_FUSION_PATH
pip install -e .
```

## Run
### Profile time for all cases
```bash
cd $OPACUS_FUSION_PATH/examples
python benchmark_scripts/profile_time_all.py
```

### Profile time
```bash
cd $OPACUS_FUSION_PATH/examples
python benchmark.py --input_size 32 --model_type cnn --architecture resnet18 --dpsgd_mode naive --batch_size 16 --profile_time # DPSGD
python benchmark.py --input_size 32 --model_type cnn --architecture resnet18 --dpsgd_mode reweight --batch_size 16 --profile_time # DPSGD(R)
python benchmark.py --input_size 32 --model_type cnn --architecture resnet18 --dpsgd_mode elegant --batch_size 16 --profile_time # Proposed
```

### Profile memory
```bash
cd $OPACUS_FUSION_PATH/examples
python benchmark.py --input_size 32 --model_type cnn --architecture resnet18 --dpsgd_mode naive --batch_size 16 --profile_memory --warm_up_steps 0 --steps 1 # DPSGD
python benchmark.py --input_size 32 --model_type cnn --architecture resnet18 --dpsgd_mode reweight --batch_size 16 --profile_memory --warm_up_steps 0 --steps 1 # DPSGD(R)
python benchmark.py --input_size 32 --model_type cnn --architecture resnet18 --dpsgd_mode elegant --batch_size 16 --profile_memory --warm_up_steps 0 --steps 1 # Proposed
```

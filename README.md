## Installation
1. Install torch from https://pytorch.org/get-started/locally/
2. Download opacus-fusion from https://github.com/parkbeomsik/opacus-fusion
3. Download cutlass from https://github.com/parkbeomsik/cutlass
4. Install cutlass_wgrad_grouped (It will build lib and include files in subdirectory)
```bash
cd $WHERE_YOU_DOWNLOAD_OPACUS_FUSION
cd cutlass_wgrad_grouped
mkdir build && cd build
cmake .. -DCUTLASS_PATH=$WHERE_YOU_DOWNLOAD_CUTLASS
make install
```
5. Install opacus-fusion
```bash
cd $WHERE_YOU_DOWNLOAD_OPACUS_FUSION
pip install -e .
```
6. Install grad_example_module
```bash
cd $WHERE_YOU_DOWNLOAD_OPACUS_FUSION
cd grad_example_module
python setup.py install
```
7. Install custom_rnn
```bash
cd $WHERE_YOU_DOWNLOAD_OPACUS_FUSION
cd custom_rnn
python setup.py install
```

## Run
```bash
cd examples
python benchmark_scripts/profile_time_all.py
```
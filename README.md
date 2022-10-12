<p align="center"><img src="https://github.com/pytorch/opacus/blob/main/website/static/img/opacus_logo.svg" alt="Opacus" width="500"/></p>

<hr/>

[![CircleCI](https://circleci.com/gh/pytorch/opacus.svg?style=svg)](https://circleci.com/gh/pytorch/opacus)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/license-apache2-green.svg)](LICENSE)

[Opacus](https://opacus.ai) is a library that enables training PyTorch models with differential privacy.
It supports training with minimal code changes required on the client, has little impact on training performance, and allows the client to online track the privacy budget expended at any given moment.

## Target audience
This code release is aimed at two target audiences:
1. ML practitioners will find this to be a gentle introduction to training a model with differential privacy as it requires minimal code changes.
2. Differential Privacy researchers will find this easy to experiment and tinker with, allowing them to focus on what matters.


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

To train your model with differential privacy, all you need to do is to instantiate a `PrivacyEngine` and pass your model, data_loader, and optimizer to the engine's `make_private()` method to obtain their private counterparts.

```python
# define your components as usual
model = Net()
optimizer = SGD(model.parameters(), lr=0.05)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1024)

# enter PrivacyEngine
privacy_engine = PrivacyEngine()
model, optimizer, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=data_loader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
)
# Now it's business as usual
```

The [MNIST example](https://github.com/pytorch/opacus/tree/main/examples/mnist.py) shows an end-to-end run using Opacus. The [examples](https://github.com/pytorch/opacus/tree/main/examples/) folder contains more such examples.

### Migrating to 1.0

Opacus 1.0 introduced many improvements to the library, but also some breaking changes.
If you've been using Opacus 0.x and want to update to the latest release,
please use this [Migration Guide](https://github.com/pytorch/opacus/blob/main/Migration_Guide.md)


## Learn more

### Interactive tutorials

We've built a series of IPython-based tutorials as a gentle introduction to training models
with privacy and using various Opacus features.

- [Building an Image Classifier with Differential Privacy](https://github.com/pytorch/opacus/blob/main/tutorials/building_image_classifier.ipynb)
- [Training a differentially private LSTM model for name classification](https://github.com/pytorch/opacus/blob/main/tutorials/building_lstm_name_classifier.ipynb)
- [Building text classifier with Differential Privacy on BERT](https://github.com/pytorch/opacus/blob/main/tutorials/building_text_classifier.ipynb)
- [Opacus Guide: Introduction to advanced features](https://github.com/pytorch/opacus/blob/main/tutorials/intro_to_advanced_features.ipynb)
- [Opacus Guide: Grad samplers](https://github.com/pytorch/opacus/blob/main/tutorials/guide_to_grad_sampler.ipynb)
- [Opacus Guide: Module Validator and Fixer](https://github.com/pytorch/opacus/blob/main/tutorials/guide_to_module_validator.ipynb)

## Technical report and citation
The technical report introducing Opacus, presenting its design principles, mathematical foundations, and benchmarks can be found [here](https://arxiv.org/abs/2109.12298).

Consider citing the report if you use Opacus in your papers, as follows:
```
@article{opacus,
  title={Opacus: {U}ser-Friendly Differential Privacy Library in {PyTorch}},
  author={Ashkan Yousefpour and Igor Shilov and Alexandre Sablayrolles and Davide Testuggine and Karthik Prasad and Mani Malek and John Nguyen and Sayan Ghosh and Akash Bharadwaj and Jessica Zhao and Graham Cormode and Ilya Mironov},
  journal={arXiv preprint arXiv:2109.12298},
  year={2021}
}
```

### Blogposts and talks

If you want to learn more about DP-SGD and related topics, check out our series of blogposts and talks:

- [Differential Privacy Series Part 1 | DP-SGD Algorithm Explained](https://medium.com/pytorch/differential-privacy-series-part-1-dp-sgd-algorithm-explained-12512c3959a3)
- [Differential Privacy Series Part 2 | Efficient Per-Sample Gradient Computation in Opacus](https://medium.com/pytorch/differential-privacy-series-part-2-efficient-per-sample-gradient-computation-in-opacus-5bf4031d9e22)
- [PriCon 2020 Tutorial: Differentially Private Model Training with Opacus](https://www.youtube.com/watch?v=MWPwofiQMdE&list=PLUNOsx6Az_ZGKQd_p4StdZRFQkCBwnaY6&index=52)
- [Differential Privacy on PyTorch | PyTorch Developer Day 2020](https://www.youtube.com/watch?v=l6fbl2CBnq0)
- [Opacus v1.0 Highlights | PyTorch Developer Day 2021](https://www.youtube.com/watch?v=U1mszp8lzUI)


## FAQ
Check out the [FAQ](https://opacus.ai/docs/faq) page for answers to some of the most frequently asked questions about differential privacy and Opacus.

## Contributing
See the [CONTRIBUTING](https://github.com/pytorch/opacus/tree/main/CONTRIBUTING.md) file for how to help out.
Do also check out the README files inside the repo to learn how the code is organized.

## License
This code is released under Apache 2.0, as found in the [LICENSE](https://github.com/pytorch/opacus/tree/main/LICENSE) file.

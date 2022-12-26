# PyTorch Benchmark

This script defines a simple convolutional neural network (ConvNet) in PyTorch, moves the model to the GPU, saves and loads the model, and runs a benchmark to measure the average elapsed time for running the model on the GPU.

## Requirements

To run this script, you will need to have the following packages installed:

- PyTorch: You can install PyTorch by following the instructions on the [PyTorch website](https://pytorch.org/). Make sure to install the correct version for your system (e.g., CPU-only, CUDA-enabled).

- tqdm: You can install tqdm by running `pip install tqdm` in your terminal.

Note: You may also need to have a CUDA-compatible GPU and the relevant drivers installed in order to run the script. If you do not have a CUDA GPU, you can still run the script by commenting out the lines that move the model and input data to the GPU.

## Usage

To run the script, simply execute it in your terminal:


`python pytorch_benchmark.py`

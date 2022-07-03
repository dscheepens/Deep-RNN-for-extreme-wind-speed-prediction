

# ConvLSTM-Pytorch

This code has been modified and extended from https://github.com/jhhuang96/ConvLSTM-PyTorch. 

The ConvLSTM has been proposed in the following paper: [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214)

## Experiments with ConvLSTM

The ConvLSTM network has an encoder-decoder structure, taking in a sequence of 12 frames and attempts to best predict the next 12 frames. 

Parameters can be selected as arguments in main.py. Root must be set manually. Data root is assumed to be root + '\data'. 

## Instructions

Requires `Pytorch v1.1` or later (and GPUs)

To run the ConvLSTM model:

```python
python main.py
```

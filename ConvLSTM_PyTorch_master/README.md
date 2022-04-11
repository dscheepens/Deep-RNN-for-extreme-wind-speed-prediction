

# ConvLSTM-Pytorch

## ConvRNN cell

This code has been modified and extended from https://github.com/jhhuang96/ConvLSTM-PyTorch.git. 

The ConvLSTM has been proposed in the following paper: [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214)

## Experiments with ConvLSTM

Encoder-decoder structure. Takes in a sequence of 12 wind speed fames and attempts to best predict the next 12 frames. Other options can be selected as arguments in main.py. 

## Instructions

Requires `Pytorch v1.1` or later (and GPUs)

To run the ConvLSTM model:

```python
python main.py
```

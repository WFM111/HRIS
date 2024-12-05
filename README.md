#PAPER CODE WITH HRIS
# HRIS

A robust high-resolution remote sensing image steganography network based on attention mechanisms

## Dependencies and Installation

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux)).
- [PyTorch = 1.0.1](https://pytorch.org/) .

## Get Started



- Run `python TRAIN.py` for training.

- Set the model path (where the trained model saved) and the image path (where the image saved during testing) to your local path.

  `line126:  '--train_set' default= '' `

  `line128:'--val_set'  default= '' `

  `line133:  '--load_weights' default= '' `

  

- Run `python TEST.py` for testing.

- Set the model path (where the trained model saved) and the image path (where the image saved during testing) to your local path.

  `line91:  WEIGHT_PATH = '' `#The parameter file for your first step of training.

  `line92:  WEIGHT_PATH1 = '' `#The parameter file for your finally step of training.

  `line93:  test_cover_PATH = '' `

  `line94:  test_secret_PATH = '' `

```

```

## Trained Model



- Here we provide a trained model
- link: https://pan.baidu.com/s/1Uyf1cPfXGkn0YWVmuiHIRA Extracted code: 2as8 
- link: https://pan.baidu.com/s/1_elwOInT3Jo68R-oWnMVTA Extracted code: m4it
- Fill in the `MODEL_PATH` and `MODEL_PATH1` before testing by the trained model.

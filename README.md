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
- 链接: https://pan.baidu.com/s/1Uyf1cPfXGkn0YWVmuiHIRA 提取码: 2as8 复制这段内容后打开百度网盘手机App，操作更方便哦
- 链接: https://pan.baidu.com/s/1_elwOInT3Jo68R-oWnMVTA 提取码: m4it 复制这段内容后打开百度网盘手机App，操作更方便哦
- Fill in the `MODEL_PATH` and `MODEL_PATH1` before testing by the trained model.

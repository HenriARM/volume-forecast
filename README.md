# volume-forecast
BTC/USDT Volume Forecast with Machine Learning

# Setup:
Install conda and create new environment:
```.bash
$ conda create --name=volfor python=3.9
$ conda activate volfor
(volfor) $ pip install -r requirements.txt
```

Install `.pkl` dataset files from [drive](https://drive.google.com/drive/folders/19DESpljw9d0EcUI0PVQ2CfvX2l5sRuZv?usp=sharing)

# Project structure:
* `datasets.py` - creates dataset
* `hypergbm.py` - runs HyperGBM to train decision trees + inference
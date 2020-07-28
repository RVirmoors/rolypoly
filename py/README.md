# rolypoly~ Python implementation
work in progress

tested on Windows 10 x64 w/ cpu & Google Colab w/cuda

## build (Win)

    pip install -r packages.txt
    pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
(see [pytorch get-started](https://pytorch.org/get-started/locally/))

for pretraining, download Groove MIDI Dataset from [Magenta](https://magenta.tensorflow.org/datasets/groove#download)
and place it in

    data/

(this will create `data/groove/') then run:

    python train_gmd.py --source midi

to build the .csv files to be used for training.
You may then zip your `groove` folder and train it in this [Google Colab](https://colab.research.google.com/drive/1t5SOnI0lW-XssYXgDfp7iXeQG4xt47ZT?usp=sharing) notebook,
or just follow the steps in the notebook locally.

## training

to interactively train your model, open the Max patch in `../max/roly-py-basic.maxpat` and then do:

    python roly.py

(will automatically preload `models/last.pt`) and save your performance. Then:

    python roly.py --offline

(will automatically use `takes/last.csv`) to train and then save your new model to `models/last.pt`

annnnd repeat.

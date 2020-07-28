# rolypoly~ Python implementation
work in progress

tested on Windows 10 x64, cpu & cuda

(OSX version coming soon)


## build (Win)

    pip install -r packages.txt
    pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
(see [pytorch get-started](https://pytorch.org/get-started/locally/))

for pretraining, download Groove MIDI Dataset from [Magenta](https://magenta.tensorflow.org/datasets/groove#download)
and place it in
    \data\

## training

to interactively train your model, do:
    python roly.py

(will automatically preload `models/last.pt`) and save your performance. Then:
    python roly.py --offline

(will automatically use `takes/last.csv`) to train and then save your new model to `models/last.pt`

annnnd repeat.

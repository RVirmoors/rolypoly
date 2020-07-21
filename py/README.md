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

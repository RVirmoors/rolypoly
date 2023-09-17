# rolypoly~

a Max object that 
- listens to your audio instrument
- interprets a MIDI drum track in anticipation and reaction to your expressive nuances
- is able to generate drum hits on the fly, autoregressively

version 2.0.1, August 2023, experimental build // [version history](VERSIONS.md)

## quickstart

first install the [FluCoMa](https://www.flucoma.org/) package - you can find it in Max's package manager

get the package from the [Releases](https://github.com/RVirmoors/rolypoly/releases) tab, and extract it into your `Documents/Max 8` directory

- if using Windows, [download Libtorch (Release version)](https://pytorch.org/get-started/locally/) and extract all the .dll files from `libtorch/lib` to the /`rolypoly/support` directory

open the rolypoly~ overview patch from the `Extras` menu in Max

demo video coming very soon!

## how does it work?

see my [demo paper](https://aimc2023.pubpub.org/pub/ud9m40jc) "Finetuning Rolypoly~ 2.0: an expressive drum machine that adapts with every performance" presented at AIMC2023

i'm also preparing a blog covering hands-on specifics

and a series of [how-to videos](https://youtube.com/playlist?list=PLkr4iJAO7fYSMZM1oYECK5GKXrWN6zdq1) on related topics

![Animated workflow diagram](_assets/workflow.gif)

## pretraining your own model

clone into `Max 8/Packages`, fetching submodules: 
- `git clone https://github.com/RVirmoors/rolypoly --recursive`

to parse the GMD dataset (or your own, similarly formatted):

- install the required Python packages: `torch, pretty_midi, numpy`
- download GMD and extract to `py/data/groove`, then use this Python script to parse .mid tracks into .csv data files

```
cd py
python midi_to_csv.py
```
this generates .csv files for all relevant .mid files in info.csv

then build the pretrain executable (windows, for now):
```
cd ../source/projects/pretrain
cmake --build . --config Release
```

if CMake doesn't do it, then manually copy the .dll files from `libtorch/lib` next to the newly-generated `pretrain.exe`. Also move the `groove` folder next to `pretrain.exe`, and run:

```
pretrain
```

## build the Max object from source (Win or MacOS)

you need [CMake](https://cmake.org/download/) installed

create a folder called `libtorch` and [download+extract LibTorch](https://pytorch.org/get-started/locally/) (Release version) into it
- the MacOS download is Intel-only; for M1 (ARM) chips you can find working build [here](https://github.com/mlverse/libtorch-mac-m1/releases/tag/LibTorch).

you tree should look like this:
```
- libtorch
    - bin, ...
- rolypoly
    - _assets, ...
```

now go back into `rolypoly`, create a `build` subfolder and enter it:

```
mkdir build
cd build
```
then run:
- on Windows (64 bit): 

```
cmake . -S ..\source\projects\rolypoly_tilde  -DCMAKE_BUILD_TYPE:STRING=Release -A x64  -DTorch_DIR="..\..\libtorch\share\cmake\Torch"
```

- on MacOS (w/ M1 ARM arch)

```
cmake ../source/projects/rolypoly_tilde  -DCMAKE_BUILD_TYPE=Release -DTorch_DIR='/Users/rv/Documents/GitHub/libtorch/share/cmake/Torch' -DCMAKE_OSX_ARCHITECTURES=arm64;
```

- on MacOS (w/ Intel arch)

```
cmake ../source/projects/rolypoly_tilde  -DCMAKE_BUILD_TYPE=Release -DTorch_DIR='/Users/rv/Documents/GitHub/libtorch/share/cmake/Torch'
```

and finally:

```
cmake --build . --config Release
```

## say hi

if you're interested in this tech and would like to use it / comment / contribute, I want to hear from you! Open an issue here or contact me: `grigore dot burloiu at unatc dot ro`

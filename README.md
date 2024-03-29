# rolypoly~

a Max object that 
- listens to your audio instrument
- interprets a MIDI drum track in anticipation and reaction to your expressive nuances
- is able to generate drum hits on the fly, autoregressively

version 0.2.11, Sep 2023 | [version history](VERSIONS.md)

## quickstart

first install the [FluCoMa](https://www.flucoma.org/) package - you can find it in Max's package manager

get the package from the [Releases](https://github.com/RVirmoors/rolypoly/releases) tab, and extract it into your `Documents/Max 8` directory

- if using Windows, [download Libtorch (Release version)](https://pytorch.org/get-started/locally/) and extract all the .dll files from `libtorch/lib` to the `rolypoly/support` directory
    - if you have an NVIDIA CUDA-enabled GPU, choose the CUDA version of LibTorch

open the rolypoly~ overview patch from the `Extras` menu in Max

demo video coming very soon!

## how does it work?

see my [demo paper](https://aimc2023.pubpub.org/pub/ud9m40jc) "Finetuning Rolypoly~ 2.0: an expressive drum machine that adapts with every performance" presented at AIMC2023

i'm also doing [blogs](https://rvirmoors.github.io/2023/09/16/rolypoly-aimc/) and [how-to videos](https://youtube.com/playlist?list=PLkr4iJAO7fYSMZM1oYECK5GKXrWN6zdq1) on related topics

![Animated workflow diagram](_assets/workflow.gif)

## pretraining your own model

this should work on Mac but I'd only recommend trying it on a PC w/ CUDA:

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

then build the pretrain executable:
```
cd ../source/projects/pretrain
cmake --build . --config Release
```

if CMake doesn't do it, then manually copy the .dll files from `libtorch/lib` next to the newly-generated `pretrain.exe`. Also move the `groove` folder next to `pretrain.exe`, and run:

```
pretrain
```

## build the Max object from source (Win or Mac)

you need [CMake](https://cmake.org/download/) installed

create a folder called `libtorch` and [download+extract LibTorch](https://pytorch.org/get-started/locally/) (Release version) into it
- the provided MacOS download is Intel-only; for ARM (M1) chips you can find a working build [here](https://github.com/mlverse/libtorch-mac-m1/releases/tag/LibTorch).
- since we don't yet have a [Universal (FAT)](https://developer.apple.com/documentation/apple-silicon/porting-your-macos-apps-to-apple-silicon#Obtain-Universal-Versions-of-Linked-Libraries) LibTorch library, we must compile Intel and ARM binaries separately
- if you're on an Intel Mac, please name your folder `libtorch_x86`

your tree should look like this:
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
cmake ..\source\projects\rolypoly_tilde  -DCMAKE_BUILD_TYPE:STRING=Release -A x64  -DTorch_DIR="..\..\libtorch\share\cmake\Torch"
```

- on MacOS (w/ M1 ARM arch)

```
cmake ../source/projects/rolypoly_tilde  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64;
```

- on MacOS (w/ x86 Intel arch)

```
cmake ../source/projects/rolypoly_tilde  -DCMAKE_BUILD_TYPE=Release 
```

and finally:

```
cmake --build . --config Release
```

Once the object is built, if it (and the torch .dll files, on Windows) are in Max's path, you should be able to see the object. You still need roly.pt from the release package (or your own pretraining, see above).

Troubleshooting: if you get an Error 242 message, you might need to delete/rename a .dll file, as mentioned [here](https://stackoverflow.com/questions/62961170/why-do-i-get-the-error-the-ordinal-242-could-not-be-located-in-the-dynamic-link). I've also had to delete this one: `C:\Program Files (x86)\Common Files\Intel\Shared Files\cpp\Bin\Intel64`

## say hi

if you're interested in this tech and would like to use it / comment / contribute, I want to hear from you! Open an issue/pull request, or contact me: `grigore dot burloiu at unatc dot ro`

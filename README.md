# rolypoly~

a Max object that 
- listens to your audio instrument
- interprets a MIDI drum track in anticipation and reaction to your expressive nuances

version 2.0.1, August 2023, experimental build.

## quickstart

get the package from the Releases tab, and extract it into your `Documents/Max 8` folder

[download Libtorch (Release version)](https://pytorch.org/get-started/locally/) and extract all the .dll files from `libtorch/lib` to `c:\Program Files\Cycling '74\Max 8\resources\support\` *(or the /support directory in your package)*

run `help/rolypoly.maxhelp` for the basics

## how does it work?

see my paper "Finetuning Rolypoly~ 2.0: an expressive drum machine that adapts with every performance" presented at AIMC2023

i'm also writing a blog covering hands-on specifics

and a series of [how-to videos](https://youtube.com/playlist?list=PLkr4iJAO7fYSMZM1oYECK5GKXrWN6zdq1)

## training your own model

clone into `Max 8/Packages`, fetching submodules: 
- `git clone https://github.com/RVirmoors/rolypoly --recursive`

to parse the GMD dataset (or your own, similarly formatted):

- download GMD and extract to `pretrain/build/Release/groove`

- use this Python script to parse .mid tracks into .csv data files


```
cd source/projects/pretrain/build
cmake --build . --config Release
cd Release
```

run:
```
pretrain
```


## build the Max object from source (windows)

you need [CMake](https://cmake.org/download/) installed

create a subfolder called `libtorch` and[download+extract LibTorch](https://pytorch.org/get-started/locally/) (Release version) into it

now go back to the project root, create a `build` subfolder and enter it:
- `cd ../..`
- `mkdir build`
- `cd build`
inside it, run:
- `cmake . -S ..\source\projects\rolypoly_tilde  -DCMAKE_BUILD_TYPE:STRING=Release -A x64  -DTorch_DIR="..\libtorch\share\cmake\Torch"`
- `cmake --build . --config Release`

## say hi

if you're interested in this tech and would like to use it / comment / contribute, I want to hear from you! Open an issue here or contact me: `grigore dot burloiu at unatc dot ro`

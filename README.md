# rolypoly~

a Max object that listens to your audio instrument and interprets a MIDI drum track in anticipation and reaction to your expressive nuances

- version 2.0.1, August 2023

## quickstart

get the package from the Releases tab, and extract it into your `Documents/Max 8` folder

[download Libtorch (Release version)](https://pytorch.org/get-started/locally/) and extract all the .dll files from `libtorch/lib` to `c:\Program Files\Cycling '74\Max 8\resources\support\` *(or the /support directory in your package)*

run `help/rolypoly.maxhelp` for the basics

## how does it work?

see my paper "Finetuning Rolypoly~ 2.0: an expressive drum machine that adapts with every performance" presented at AIMC2023

i'm also writing a blog covering some specifics

and coming soon a series of how-to videos, hopefully

## training your own model

clone into `Max 8/Packages`, fetching submodules: 
- `git clone https://github.com/RVirmoors/rolypoly --recursive`

(optional but recommended) create a new virtual environment:
- `python -m venv venv`

you need the following Python modules:
- pytorch (I use `ltt` to get it, but you can go [the classic way](https://pytorch.org/get-started/locally/)):
```
python -m pip install light-the-torch
ltt install torch
```
- others:
`pip install numpy`

make sure you're in the `py` folder:
- `cd py`


...


```
cd source/projects/pretrain/build
cmake --build . --config Release
cd Release
```
- download GMD and extract to `pretrain/build/Release/groove`
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
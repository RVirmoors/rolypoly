# rolypoly~

work underway (Jan-Feb '22) on porting to ircam's nn~ for real-time use in Max!

## quickstart

(coming soon) get the package from the Releases tab

## training your own model

clone into `Max 8/Packages`, fetching submodules: 
- `git clone https://github.com/RVirmoors/rolypoly --recursive`

(optional but recommended) create a new virtual environment:
- `python -m venv venv`

you need the following Python modules:
- pytorch (I use `ltt` to get it, but you can go [the clasic way](https://pytorch.org/get-started/locally/)):
```
python -m pip install light-the-torch
ltt install torch
```
- nn_tilde:
`pip install nn_tilde`
- others:
`pip install numpy`

make sure you're in the `py` folder:
- `cd py`

and run the train script to train on your data (coming soon)

and then export it to a `.ts` file which `rolypoly~` can use in Max:
- `python export.py`


## build the Max object from source (windows)

you need [CMake](https://cmake.org/download/) installed

create a subfolder called `libtorch` and[download+extract LibTorch](https://pytorch.org/get-started/locally/) into it

copy the libtorch *.dll files from `libtorch/lib` to `c:\Program Files\Cycling '74\Max 8\resources\support\` *(or the /support directory in your package)*

first you need to build `nn_tilde` (just the backend is enough): inside `nn_tilde/` create a `build` subfolder and enter it:
- `cd nn_tilde`
- `mkdir build`
- `cd build`
inside it, run:
- `cmake . -S ..\src\backend  -DCMAKE_BUILD_TYPE:STRING=Release -A x64 -DTorch_DIR="..\..\libtorch\share\cmake\Torch"` (if this Torch_DIR doesn't work, replace it with the absolute path)
- `cmake --build . --config Release`

now go back to the project root, create a `build` subfolder and enter it:
- `cd ../..`
- `mkdir build`
- `cd build`
inside it, run:
- `cmake . -S ..\source\projects\rolypoly_tilde  -DCMAKE_BUILD_TYPE:STRING=Release -A x64  -DTorch_DIR="..\libtorch\share\cmake\Torch"`
- `cmake --build . --config Release`

## say hi

if you're interested in this tech and would like to use it / comment / contribute, I want to hear from you! Open an issue here or contact me [@growlerpig](https://twitter.com/growlerpig/) / `grigore dot burloiu at unatc dot ro`

# rolypoly~

work underway (Jan-Feb '22) on porting to ircam's nn~ for real-time use in Max!

## quickstart

(coming soon) get the package from the Releases tab

## training your own model

clone into `Max 8/Packages`, fetching submodules: `git clone https://github.com/RVirmoors/rolypoly --recursive`

(optional but recommended) create a new virtual environment
- `python -m venv venv`

you need the following Python modules:
- pytorch
```
python -m pip install light-the-torch
ltt install torch
```
- nn_tilde
`pip install nn_tilde`
- others
`pip install numpy`

make sure you're in the `py` folder
- `cd py`

and run the train script to train on your data (coming soon)

and then export it to a `.ts` file which `rolypoly~` can use in Max
- `python export.py`


## build from source (windows)

create a subfolder called `libtorch` and[download+extract LibTorch](https://pytorch.org/get-started/locally/) into it

copy the libtorch *.dll files from `libtorch/lib` to `c:\Program Files\Cycling '74\Max 8\resources\support\` *(or the /support directory in your package)*

go to `build/` and run:
- `cmake . -S ..\source\projects\rolypoly_tilde  -DCMAKE_BUILD_TYPE:STRING=Release -A x64  -DTorch_DIR="..\libtorch\share\cmake\Torch"`
- `cmake --build . --config Release`

## say hi

if you're interested in this tech and would like to use it / comment / contribute, I want to hear from you! Open an issue here or contact me [@growlerpig](https://twitter.com/growlerpig/) / `grigore dot burloiu at unatc dot ro`

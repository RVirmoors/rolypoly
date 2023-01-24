# rolypoly~

currently works as a Python + Max ensemble (see [/py](/py) folder) - [intro video](https://youtu.be/UHBIzfc5DCI)

work underway (Jan-Feb '22) on porting to ircam's nn~ for real-time use in Max!

## build from source

clone into `Max 8/Packages`, fetching submodules: `git clone https://github.com/RVirmoors/rolypoly --recursive`

download LibTorch into `(Project_Dir)/libtorch`

copy the libtorch *.dll files to `c:\Program Files\Cycling '74\Max 8\resources\support\` *(or the /support directory in your package)*

go to `build/` and run:
- `cmake . -S ..\source\projects\rolypoly_tilde  -DCMAKE_BUILD_TYPE:STRING=Release -A x64  -DTorch_DIR="..\libtorch\share\cmake\Torch"`
- `cmake --build . --config Release`

## say hi

if you're interested in this tech and would like to use it / comment / contribute, I want to hear from you! Open an issue here or contact me [@growlerpig](https://twitter.com/growlerpig/) / `grigore dot burloiu at unatc dot ro`

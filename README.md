# Video Stabilization

### Prerequisites
Some features of our program (e.g. `OpticalFlow` calculation) requires **OpenCV's** CUDA functions. So you should install **OpenCV** with **CUDA** support.

### Compilation & usage
Program compiles with CMake:
```
mkdir build
cd build
cmake ..
make
```

After this, in **build** directory 4 executables will be generated.

The first one is video stabilization program itself, which can be run to stabilize a video. You can run it as follows: `./vidstab </path/to/original_video.mp4>`. Stabilized video will be written to file **./output.mp4**. You will see progress as program runs. Note that video inpainting takes a lot of time (for example for 4 sec video of size 300x300 it took about 2 min). Source code for this target can be found in **src/main.cpp** file (it uses **src/adapted_optical_flow.hpp**)

Second executable is `psnr` which evaluates quality of a video. You can run it as follows: `../videos/car_original_resized.mp4 ../videos/car_online_stabilazer_resized.mp4 10000 1`. Source code can be found in **src/psnr.cpp**

Third and fourth executables are related to ECC alignment algorithm. They are `ECCTest` and `OurECC`. Source code -- **src/ourECC\*.cpp**

### Results
Presentation, result description and implementation are explained in this [YouTube video](https://youtu.be/tfT4fq6IhhI) (accessible only from **@ucu.edu.ua email**)

A detailed explanations on how it works and principles under the hood can by found in our [Report](Report.pdf)

You can also find original video, stabilized without inpainting, stabilized with inpainting videos and stabilized by online stabilizer videos in **videos** folder. Names are self-explanatory.

### Authors
* Bohdan Mahometa
* Yaroslav Revera
* Maksym Tsapiv

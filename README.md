# repeater-max-cuda
The CUDA version of the Repeater MAX. Repeats intentions up to 1.5+ Billion times per second.

Requires Visual Studio 2019 Community: https://visualstudio.microsoft.com/downloads/
and CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit

To compile: nvcc intention_repeater_max_cuda.cu -O 3 -o "intention_repeater_max_cuda.exe"

Special thanks to Karteek Sheri for providing the CUDA functionality.

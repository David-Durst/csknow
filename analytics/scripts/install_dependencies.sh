sudo apt install -y gcc g++ gdb libssl-dev make libopencv-dev python3-opencv cmake
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.13.0+cpu.zip -d external/
rm libtorch-cxx11-abi-shared-with-deps-1.13.0+cpu.zip
git submodule update --init --recursive

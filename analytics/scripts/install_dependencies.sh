sudo apt install -y gcc g++ gdb libssl-dev make libopencv-dev python3-opencv cmake libhdf5-dev
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.13.1+cpu.zip -d external/
rm libtorch-cxx11-abi-shared-with-deps-1.13.1+cpu.zip
git submodule update --init --recursive

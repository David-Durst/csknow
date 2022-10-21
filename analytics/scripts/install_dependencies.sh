sudo apt install -y gcc g++ libssl-dev make libopencv-dev python3-opencv cmake
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.12.1+cpu.zip -d external/
rm libtorch-cxx11-abi-shared-with-deps-1.12.1+cpu.zip

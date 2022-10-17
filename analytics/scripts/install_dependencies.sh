sudo apt install gcc g++ libssl-dev make
sudo snap install cmake --classic
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.12.1%2Bcpu.zip
unzip libtorch-shared-with-deps-1.12.1+cpu.zip -d external/
rm libtorch-shared-with-deps-1.12.1+cpu.zip

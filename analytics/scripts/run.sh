mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
./csknow /home/durst/big_dev/csknow/local_data 

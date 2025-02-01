export PATH=$PATH:~/intel/oneapi/2025.0/bin/
rm -r ./build
cmake -DCMAKE_PREFIX_PATH=~/.local/lib -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Profile -S . -B ./build
cmake --build ./build -j
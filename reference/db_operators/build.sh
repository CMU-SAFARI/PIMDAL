rm -r ./build
cmake -DCMAKE_PREFIX_PATH=~/.local/lib -DCMAKE_BUILD_TYPE=Release -S . -B ./build
cmake --build ./build -j
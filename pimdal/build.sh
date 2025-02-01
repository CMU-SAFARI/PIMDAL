source /upmem-sdk-2023.2.0/upmem_env.sh
cmake -DCMAKE_BUILD_TYPE=Release -S . -B ./build
cmake --build ./build
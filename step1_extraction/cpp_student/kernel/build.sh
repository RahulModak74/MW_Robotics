#!/usr/bin/env bash
# build.sh - compile the fly-by-wire kernel against the MuJoCo C library shipped in the mujoco wheel.
set -e
cd "$(dirname "$0")"
MJ=$(python3 -c "import mujoco, os; print(os.path.dirname(mujoco.__file__))")
LIB=$(ls $MJ/libmujoco.so.* | head -1)
EXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
g++ -O3 -march=native -mavx2 -mfma -fopenmp -DNDEBUG -DEIGEN_NO_DEBUG -shared -fPIC -std=c++17 \
    $(python3 -m pybind11 --includes) -I$MJ/include -I/usr/include/eigen3 \
    fbw_kernel.cpp -o fbw_kernel$EXT $LIB -Wl,-rpath,$MJ
echo built fbw_kernel$EXT

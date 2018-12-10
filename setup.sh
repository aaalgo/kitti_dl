#!/bin/bash
#git submodule init
#git submodule update
git clone https://github.com/aaalgo/streamer
./setup.py build
#cd kitti_native_evaluation
#cmake .
make

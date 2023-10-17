#!/bin/bash
#
include_path=$(<../../compile_flags.txt)
g++ -std=c++20 $include_path main.cpp -o main && ./main

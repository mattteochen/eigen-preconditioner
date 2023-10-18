#!/bin/bash

source=$1

include_path=$(<../../compile_flags.txt)
g++ -std=c++20 $include_path $source -o main && ./main

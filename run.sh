#!/bin/bash
#
include_path=$(<compile_flags.txt)
g++ -std=c++20 $include_path custom_preconditioner_base.cpp -o custom_preconditioner_base && ./custom_preconditioner_base

#!/bin/bash

g++ -std=c++20 -I ${mkEigenInc} main.cpp -o main && ./main

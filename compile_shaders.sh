#!/bin/env sh
cd "$(dirname "$(realpath -- "$0")")";

glslc tutorial.frag -o tutorial_frag.spv
glslc tutorial.vert -o tutorial_vert.spv

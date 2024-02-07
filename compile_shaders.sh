#!/bin/env sh
cd "$(dirname "$(realpath -- "$0")")";

glslc shaders/tutorial.frag -o shaders/tutorial_frag.spv
glslc shaders/tutorial.vert -o shaders/tutorial_vert.spv

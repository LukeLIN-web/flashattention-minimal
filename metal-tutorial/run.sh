#!/bin/bash
rm add.air default.metallib VectorAdd
xcrun -sdk macosx metal -c add.metal -o add.air
xcrun -sdk macosx metallib add.air -o default.metallib
clang -fobjc-arc -framework Metal -framework Foundation  -framework CoreGraphics main.m MetalAdder.m -o VectorAdd
echo "Compiled Done"
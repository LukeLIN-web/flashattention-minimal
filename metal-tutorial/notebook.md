# note


`id<MTLDevice> device = MTLCreateSystemDefaultDevice(); ` cannot find, why?



1. xcrun: error: unable to find utility "metal", not a developer tool or in PATH? 

download Xcode.

```bash
#!/bin/bash
xcrun -sdk macosx metal -c add.metal -o add.air
xcrun -sdk macosx metallib add.air -o default.metallib
clang -fobjc-arc -framework Metal -framework Foundation  -framework CoreGraphics  main.m MetalAdder.m -o VectorAdd
```
2. 2024-08-11 20:26:35.885 VectorAdd[66115:1733085] Failed to find the default library. how to solve?

https://developer.apple.com/documentation/metal/1433401-mtlcreatesystemdefaultdevice    -framework CoreGraphics 




This is but a simple bunch of code to test performance of widely known convolutional neural networks on several publicly available image datasets.

## Building & running

Just build cuda-convnet included as git submodule & run run.sh.
Dont forget to fix its CMakeLists.txt:52 by uncommenting `arch=compute_20,code=sm_21` for older video cards (like GTX 450 of mine :unamused:) & to pass something like
```
-DBLAS_LIBRARIES=/usr/lib/atlas-base/libcblas.so
```
to cmake when building.

Also, several additional modifications to python code of cuda-convnet would be required.
I'll describe them later.


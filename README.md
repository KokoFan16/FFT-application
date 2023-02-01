# FFT-application
## install FFTW3
```
$ ./configure --enable-mpi --prefix=/somewhere/else/than/usr/local
$ make
$ make install
```

## install example
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## run example
$ cd build/example/
$ mpirun -n 8 ./fftw_mpi_1d N (e.g.,64)

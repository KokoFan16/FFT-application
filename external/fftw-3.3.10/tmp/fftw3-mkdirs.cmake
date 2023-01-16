# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/Users/kokofan/Documents/project/FFT-application/external/fftw-3.3.10/src/fftw3"
  "/Users/kokofan/Documents/project/FFT-application/external/fftw-3.3.10/src/fftw3-build"
  "/Users/kokofan/Documents/project/FFT-application/external/fftw-3.3.10"
  "/Users/kokofan/Documents/project/FFT-application/external/fftw-3.3.10/tmp"
  "/Users/kokofan/Documents/project/FFT-application/external/fftw-3.3.10/src/fftw3-stamp"
  "/Users/kokofan/Documents/project/FFT-application/external/fftw-3.3.10/src"
  "/Users/kokofan/Documents/project/FFT-application/external/fftw-3.3.10/src/fftw3-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/kokofan/Documents/project/FFT-application/external/fftw-3.3.10/src/fftw3-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/kokofan/Documents/project/FFT-application/external/fftw-3.3.10/src/fftw3-stamp${cfgdir}") # cfgdir has leading slash
endif()

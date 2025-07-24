# SpleeterExe

A C++ application for source separation using a TensorFlow model (Spleeter 2stems) originally created for [Basset](https://github.com/mohammadmansour200/basset). This tool processes PCM audio files, extracting vocals from them.

## Features

- Fast, chunked audio processing for large files
- Uses TensorFlow C API for inference
- Outputs separated vocals in PCM format
- Cross-platform (Linux, Windows)

## Requirements

- C++17 compatible compiler (e.g., g++, MSVC)
- CMake >= 3.15 (for CMake build)
- TensorFlow C library 1.15 (provided in `third_party/tensorflow-cpu`)

> Linux http://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.15.0.tar.gz

> Windows http://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-1.15.0.zip

## File Structure

```
SpleeterExe/
├── main.cpp
├── src/spleeter/SpleeterProcessor.cpp
├── src/spleeter/include/SpleeterProcessor.hpp
├── models/2stems/
│   ├── saved_model.pb
│   └── variables/
├── third_party/tensorflow-cpu/
│   ├── include/
│   └── lib/
├── CMakeLists.txt
├── Makefile
└── README.md
```

## Build Instructions

### Using Makefile (Linux)

```sh
make
make install  # Copies required libraries and models to build/
```

- Output binary: `build/SpleeterExe`

### Using CMake (Windows)

```sh
mkdir build
cd build
cmake ..
cmake --build .
```

- Output binary: `build/SpleeterExe.exe`

## Usage

```
SpleeterExe <input_pcms_path> <output_pcms_path> <file_duration> <chunk_seconds>
```

- `input_pcms_path`: Path to input PCM (float32, stereo, 44100Hz)
- `output_pcms_path`: Base path for output PCM (vocals)
- `file_duration`: Duration of input file in seconds
- `chunk_seconds`: Chunk size in seconds (for processing only, not output)

**Example:**

```
SpleeterExe input.pcm output 180 30
```

This will process `input.pcm` (3 minutes), outputting vocals to `output.pcm`.

## Troubleshooting

- Ensure TensorFlow libraries are present in the correct location (`third_party/tensorflow-cpu/lib` or copied to build output).
- Input PCM must be float32, stereo, 44100Hz.
- If you see model not found errors, check the `models/2stems` directory.

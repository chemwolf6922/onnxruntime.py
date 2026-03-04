# onnxruntime.py

## Build steps

### Windows

1. Prepare a shell with Visual Studio 2022 tools and python enabled.
2. `pip install -r requirements-windows.txt`
3. `python get_onnxruntime.py`

```PowerShell
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Linux

1. `pip install -r requirements-linux.txt`
2. `python get_onnxruntime.py`

```bash
mkdir -p build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### macOS

1. `pip install -r requirements-macos.txt`
2. `python get_onnxruntime.py`

```bash
mkdir -p build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Example

Please install/reinstall the new wheels before running the examples.

Please find examples under `./examples`

## Excluded APIs
These APIs will not be implemented.

* AddCustomOpDomain

It's complicated to implement and does not make much sense for python inference. If possible, consider use RegisterCustomOpsLibrary instead.

* StringTensor

Need to learn more about the use cases.

## Skipped APIs
These APIs' implementation are skipped for now due to complexity or implementation order.

* OverridableInitializer

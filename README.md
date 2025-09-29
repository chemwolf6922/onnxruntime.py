# onnxruntime.py

## Build steps

### Prerequisites

1. Prepare a shell with Visual Studio 2022 tools and python enabled.
2. [optional]Create a new python environment.
3. `pip install -r requirements.txt`
4. `GetOnnxruntime.ps1`

### Build for your system

```PowerShell
mkdir build
cd build
cmake ..
cmake --build .
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

# onnxruntime.py

## Build steps

### Prerequisites

1. Prepare a shell with Visual Studio 2022 tools and python enabled.
2. [optional]Create a new python environment.
3. `pip install -r requirements.txt`
4. `GetOnnxruntime.ps1`

### Build the main package

```PowerShell
pip -m build
```

The output should be under `./dist`

### Build the resource package

```PowerShell
cd lib-package
pip -m build
```

The output should be under `./lib-package/dist`

### Build both packages

```PowerShell
./build-all.ps1
```

The output should be under `./dist`

## Example

Please install/reinstall the new wheels before running the examples.

Please find examples under `./examples`


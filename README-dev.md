# Developer notes

## Using pybind11 as a git submodule

Recommended for IDE integration and better C++/Python linting.

Commands (PowerShell):

```powershell
# Add submodule
git submodule add https://github.com/pybind/pybind11 extern/pybind11

# Initialize/Update submodules
git submodule update --init --recursive
```

CMake will automatically use `extern/pybind11` when present.

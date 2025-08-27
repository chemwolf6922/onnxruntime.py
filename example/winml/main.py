from importlib import metadata
from pathlib import Path

# DO NOT install pyort-lib for this example
# We'll use the packed one in WinML
try:
    metadata.version('pyort-lib')
    raise RuntimeError("pyort-lib is installed. Please uninstall it for this example.")
except metadata.PackageNotFoundError:
    pass

# Quirk: There is a packed msvcp140.dll in pywinrt which may conflict with other binaries.
# Thus we need to remove it.
# If you do face the missing dll issue, please install the redistributable properly from
# https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170
site_packages_path = Path(str(metadata.distribution('winrt-runtime').locate_file('')))
dll_path = site_packages_path / 'winrt' / 'msvcp140.dll'
if dll_path.exists():
    dll_path.unlink()

from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
    InitializeOptions,
    initialize
)
import winui3.microsoft.windows.ai.machinelearning as winml

class WinAppSdkHandle:
    def __init__(self, handle) -> None:
        self._handle = handle
        self._handle.__enter__()

    def __del__(self) -> None:
        self._handle.__exit__(None, None, None)

_win_app_sdk_handle = WinAppSdkHandle(initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI))

catalog = winml.ExecutionProviderCatalog.get_default()

# Quirk: Call this to create the default ort env since the WinML uses a temporary one for registering EPs.
# This will be fixed in future releases of WinML
import pyort as ort
ort.get_ep_devices()

# Install and register all compatible EPs
catalog.ensure_and_register_all_async().get()


# Normally you would import pyort here
# import pyort as ort
from argparse import ArgumentParser
import numpy as np
import tqdm
parser = ArgumentParser()
parser.add_argument("--model_path", "-m", type=Path, required=True, help="Path to the ONNX model file.")
parser.add_argument("--num_inferences", "-n", type=int, default=100, help="Number of inferences to run.")
parser.add_argument("--compile_model", "-c", action="store_true", help="Whether to compile the model.")
parser.add_argument("--ep_policy", "-p", type=str,
                    choices=list(ort.ExecutionProviderDevicePolicy.__members__.keys()), default=None,
                    help="Execution provider selection policy to use.")
parser.add_argument("--use_policy_delegate", action="store_true", help="Whether to use a custom policy delegate.")
args = parser.parse_args()

def dump_ep_device(ep_device: ort.EpDevice, index: int | None = None):
    print(f"Ep Device {index if index else ""}:")
    print(f"    EP Name:       {ep_device.ep_name}")
    print(f"    EP Vendor:     {ep_device.ep_vendor}")
    print(f"    Device Type:   {ep_device.device.type.name}")
    print(f"    Device Vendor: {ep_device.device.vendor}")

def ep_policy_delegate(ep_devices: list[ort.EpDevice],
                                model_metadata: dict[str, str],
                                runtime_metadata: dict[str, str],
                                max_eps: int) -> list[ort.EpDevice]:
    print("In EP policy delegate:")
    for (i, ep_device) in enumerate(ep_devices):
        dump_ep_device(ep_device, i)
    print(f"Model metadata: {model_metadata}")
    print(f"Runtime metadata: {runtime_metadata}")
    print(f"Max EPs allowed: {max_eps}")
    index_str = input("Select the EP device index and press Enter: ")
    index = int(index_str)
    return [ep_devices[index]]

session_options = ort.SessionOptions()
if args.ep_policy is not None:
    policy = ort.ExecutionProviderDevicePolicy.__members__[args.ep_policy]
    session_options.set_ep_selection_policy(policy)
elif args.use_policy_delegate:
    session_options.set_ep_selection_policy_delegate(ep_policy_delegate)
else:
    ep_devices = ort.get_ep_devices()
    for (i, ep_device) in enumerate(ep_devices):
        dump_ep_device(ep_device, i)
    index_str = input("Select the EP device index and press Enter: ")
    index = int(index_str)
    ep_device = ep_devices[index]
    session_options.append_execution_provider_v2([ep_device], {})

model_path = Path(args.model_path).resolve()
if args.compile_model:
    compiler = session_options.create_model_compilation_options()
    compiler.set_input_model_path(str(model_path))
    compiled_model_path = model_path.parent / (model_path.stem + "_compiled.onnx")
    if compiled_model_path.exists():
        compiled_model_path.unlink()
    compiler.compile_model_to_file(str(compiled_model_path))
    if compiled_model_path.exists():
        print(f"Compiled model saved to {compiled_model_path}")
        model_path = compiled_model_path
    else:
        print("No compile output found, using the original model.")

session = ort.Session(str(model_path), session_options)

input_info = session.get_input_info()
inputs = {}
for input_name, tensor_info in input_info.items():
    if any(dim <= 0 for dim in tensor_info.shape):
        raise RuntimeError(f"Input {input_name} has dynamic shape {tensor_info.shape}. "
                           "Please use a model with static input shape for this example.")
    inputs[input_name] = np.random.uniform(low=0 ,high=1, size=tuple(tensor_info.shape)).astype(tensor_info.dtype)

for i in tqdm.tqdm(range(args.num_inferences)):
    session.run(inputs)

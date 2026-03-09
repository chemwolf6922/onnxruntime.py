# Missing ORT C API Bindings

> **Minimum target: ORT 1.23 (API version 23).**
> APIs marked with a version require `#if ORT_API_VERSION >= N` guards when implemented.

## Currently Bound APIs

| Area | What's Covered |
|---|---|
| **Env** | `CreateEnv`, `ReleaseEnv`, `DisableTelemetryEvents`, `RegisterExecutionProviderLibrary`, `UnregisterExecutionProviderLibrary`, `GetEpDevices`, `UpdateEnvWithCustomLogLevel`, `GetNumHardwareDevices` (v24), `GetHardwareDevices` (v24) |
| **SessionOptions** | Create/Release/Clone, `SetOptimizedModelFilePath`, `SetSessionExecutionMode`, Enable/Disable Profiling, Enable/Disable MemPattern, Enable/Disable CpuMemArena, `SetSessionLogId`, `SetSessionLogVerbosityLevel`, `SetSessionLogSeverityLevel`, `SetSessionGraphOptimizationLevel`, `SetIntraOpNumThreads`, `SetInterOpNumThreads`, `RegisterCustomOpsLibrary` (V1), `RegisterCustomOpsLibrary_V2`, `RegisterCustomOpsUsingFunction`, `EnableOrtCustomOps`, `AddFreeDimensionOverride`, `AddFreeDimensionOverrideByName`, `DisablePerSessionThreads`, `AddSessionConfigEntry`, `HasSessionConfigEntry`, `GetSessionConfigEntry`, `GetSessionOptionsConfigEntries`, `SetDeterministicCompute`, `SessionOptionsSetLoadCancellationFlag`, `GetAvailableProviders`, `SessionOptionsAppendExecutionProvider` (generic key-value), `SessionOptionsAppendExecutionProvider_V2`, `SessionOptionsSetEpSelectionPolicy`, `SessionOptionsSetEpSelectionPolicyDelegate` |
| **Session** | `CreateSession`, `CreateSessionFromArray`, `Run`, `RunWithBinding`, `SessionGetInputCount/Name/TypeInfo`, `SessionGetOutputCount/Name/TypeInfo`, `SessionGetModelMetadata`, `SessionEndProfiling`, `SessionGetProfilingStartTimeNs`, `SessionGetMemoryInfoForInputs/Outputs`, `SessionGetEpDeviceForInputs/Outputs` (v24), `Session_GetEpGraphAssignmentInfo` (v24) |
| **RunOptions** | Create/Release, Set/Get RunLogVerbosityLevel, Set/Get RunLogSeverityLevel, Set/Get RunTag, Set/Unset Terminate, `AddRunConfigEntry`, `GetRunConfigEntry`, `RunOptionsAddActiveLoraAdapter` |
| **Value/Tensor** | `CreateTensorAsOrtValue`, `CreateTensorWithDataAsOrtValue`, `GetTensorMutableData`, `GetTensorTypeAndShape`, `GetTensorElementType`, `GetDimensionsCount`, `GetDimensions`, `GetSymbolicDimensions`, `ReleaseValue` |
| **TypeInfo** | `CastTypeInfoToTensorInfo`, `GetOnnxTypeFromTypeInfo`, `GetDenotationFromTypeInfo`, `GetTensorShapeElementCount`, `ReleaseTypeInfo`, `ReleaseTensorTypeAndShapeInfo` |
| **MemoryInfo** | `CreateCpuMemoryInfo`, `CreateMemoryInfo`, `MemoryInfoGetName`, `MemoryInfoGetId`, `MemoryInfoGetMemType`, `MemoryInfoGetType`, `MemoryInfoGetDeviceType`, `CompareMemoryInfo`, `ReleaseMemoryInfo` |
| **Allocator** | `GetAllocatorWithDefaultOptions` |
| **EpDevice/HardwareDevice** | `HardwareDevice_Type/VendorId/Vendor/DeviceId/Metadata`, `EpDevice_EpName/EpVendor/EpMetadata/EpOptions/Device`, `EpDevice_MemoryInfo`, `GetNumHardwareDevices` (v24), `GetHardwareDevices` (v24), `GetHardwareDeviceEpIncompatibilityDetails` (v24), `GetModelCompatibilityForEpDevices`, `GetCompatibilityInfoFromModel` (v24), `GetCompatibilityInfoFromModelBytes` (v24) |
| **ModelMetadata** | `ReleaseModelMetadata`, `ModelMetadataGetProducerName/GraphName/Domain/Description/GraphDescription/Version`, `ModelMetadataLookupCustomMetadataMap`, `ModelMetadataGetCustomMetadataMapKeys` |
| **IoBinding** | `CreateIoBinding`, `ReleaseIoBinding`, `BindInput`, `BindOutput`, `BindOutputToDevice`, `GetBoundOutputNames`, `GetBoundOutputValues`, `ClearBoundInputs`, `ClearBoundOutputs`, `SynchronizeBoundInputs`, `SynchronizeBoundOutputs` |
| **CompileApi** | `CreateModelCompilationOptionsFromSessionOptions`, `ModelCompilationOptions_SetInputModelPath`, `SetInputModelFromBuffer`, `SetOutputModelExternalInitializersFile`, `SetEpContextEmbedMode`, `SetOutputModelPath`, `SetOutputModelBuffer`, `CompileModel`, `ReleaseModelCompilationOptions`, `SetFlags`, `SetEpContextBinaryInformation`, `SetGraphOptimizationLevel` |
| **LoRA** | `CreateLoraAdapter`, `CreateLoraAdapterFromArray`, `ReleaseLoraAdapter` |

---

## Missing APIs — Inference-Related, Normal Python Use

### Group 1: Value/Tensor — Extended Operations (Medium Priority)

| C API | Description | Status | API Ver |
|---|---|---|---|
| `IsTensor` | Check if an OrtValue is a tensor | | ≤10 |
| `GetValueType` | Get ONNXType from OrtValue | | ≤10 |
| `HasValue` | Check if optional OrtValue has data | | ≤10 |
| `GetTensorMemoryInfo` | Get memory info of a tensor | | ≤10 |
| `GetTensorSizeInBytes` | Get total tensor size in bytes | | 23 |
| `FillStringTensor` | Fill a string tensor | | ≤10 |
| `GetStringTensorDataLength` | Get total string tensor byte length | | ≤10 |
| `GetStringTensorContent` | Get all string tensor content | | ≤10 |
| `GetStringTensorElementLength` | Get length of one string element | | ≤10 |
| `GetStringTensorElement` | Get one string element | | ≤10 |
| `FillStringTensorElement` | Fill one string element | | ≤10 |
| `GetResizedStringTensorElementBuffer` | Get resized buffer for a string element | | 15 |
| `GetValue` | Extract element from map/sequence OrtValue | | ≤10 |
| `GetValueCount` | Get count in map/sequence OrtValue | | ≤10 |
| `CreateValue` | Create a map/sequence OrtValue | | ≤10 |

### Group 2: TypeInfo — Map/Sequence/Optional Introspection (Medium Priority)

Needs new ORT opaque type wrappers (MapTypeInfo, SequenceTypeInfo, OptionalTypeInfo).

| C API | Description | Status | API Ver |
|---|---|---|---|
| `CastTypeInfoToMapTypeInfo` | Cast to OrtMapTypeInfo | | ≤10 |
| `CastTypeInfoToSequenceTypeInfo` | Cast to OrtSequenceTypeInfo | | ≤10 |
| `CastTypeInfoToOptionalTypeInfo` | Cast to OrtOptionalTypeInfo | | 15 |
| `GetMapKeyType` | Get map key element type | | ≤10 |
| `GetMapValueType` | Get map value type info | | ≤10 |
| `GetSequenceElementType` | Get sequence element type info | | ≤10 |
| `GetOptionalContainedTypeInfo` | Get contained type of optional | | 15 |

---

## Missing APIs — NOT Inference-Related

### Training APIs

| C API | Description |
|---|---|
| `GetTrainingApi` + all `OrtTrainingApi` functions | Full training API surface (defined in `onnxruntime_training_c_api.h`) |

---

## Missing APIs — NOT for Normal Python Use

### Initializer Management

> Skipped for now — requires exposing OrtValue as initializer inputs.

| C API | Description |
|---|---|
| `AddInitializer` | Add pre-loaded initializer |
| `AddExternalInitializers` | Replace external initializers with tensors |
| `AddExternalInitializersFromFilesInMemory` | Add external initializers from memory buffers |

### SessionOptions — Custom Logging

> Requires C callback bridging — complex and unusual need.

| C API | Description |
|---|---|
| `SetUserLoggingFunction` | Set custom user logging function |

### CompileApi — Custom Callbacks

> Requires C callback bridging.

| C API | Description |
|---|---|
| `ModelCompilationOptions_SetOutputModelWriteFunc` | Set custom write function |
| `ModelCompilationOptions_SetOutputModelGetInitializerLocationFunc` | Set initializer location function |

### Async Execution

> Requires C callback bridging. Python GIL limits benefit.

| C API | Description |
|---|---|
| `RunAsync` | Run model asynchronously |

### RunOptions — SyncStream

> Requires OrtSyncStream which isn't exposed.

| C API | Description |
|---|---|
| `RunOptionsSetSyncStream` | Set OrtSyncStream on run options |

### Raw Pointer / Unsafe APIs

> Dangerous in Python — use-after-free, no bounds checking.

| C API | Description |
|---|---|
| `GetTensorData` | Get const pointer to tensor data |
| `TensorAt` | Get pointer to a specific tensor element |
| `CreateTensorWithDataAndDeleterAsOrtValue` | Create tensor with custom deleter |

### Allocator — Extended

> Raw memory alloc/free is not for Python use.

| C API | Description |
|---|---|
| `CreateAllocator` | Create allocator from session + MemoryInfo |
| `AllocatorAlloc` | Allocate memory |
| `AllocatorFree` | Free memory |
| `AllocatorGetInfo` | Get allocator's MemoryInfo |

### Legacy Per-EP Registration

> Matching the official onnxruntime Python package: only the generic `append_execution_provider(name, options)` is exposed.
> Provider-specific V1 struct APIs and V2 Create/Update/Release APIs are used internally but not exposed as separate Python methods.

### SyncStream / Data Transfer / Shared Allocator

> Low priority, requires hardware, depends on unexposed SyncStream.

### Overridable Initializers

> Niche feature. Most models don't use overridable initializers.

| C API | Description |
|---|---|
| `SessionGetOverridableInitializerCount` | Count of overridable initializers |
| `SessionGetOverridableInitializerName` | Name of an overridable initializer |
| `SessionGetOverridableInitializerTypeInfo` | TypeInfo of overridable initializer |

### Env Extended

| C API | Description |
|---|---|---|
| `CreateEnvWithGlobalThreadPools` | Create env with global thread pools |
| `EnableTelemetryEvents` | Enable telemetry |
| `GetBuildInfoString` | Get ORT build info string |

### Custom Op Implementation

| C API | Description |
|---|---|
| `CreateCustomOpDomain` | Create custom op domain |
| `CustomOpDomain_Add` | Add custom op to domain |
| `AddCustomOpDomain` | Add custom op domain to session options |
| `CreateOpAttr`, `CreateOp`, `InvokeOp`, `CopyKernelInfo`, `ReadOpAttr` | ORT-native operator creation/invocation |
| All `KernelInfoGetAttribute_*` | Get attributes from kernel info |
| `KernelInfo_GetInputCount/OutputCount/Name/TypeInfo` | Kernel info input/output introspection |
| `KernelInfo_GetNodeName/Logger/Allocator` | Kernel info utilities |
| `KernelInfo_GetConfigEntries/OperatorDomain/Type/SinceVersion` | Kernel info config and op metadata |
| All `KernelContext_*` | Kernel context input/output/stream/allocator/logger/resource/scratch/parallel |
| All `ShapeInferContext_*` | Shape inference for custom ops |

### Execution Provider Implementation

| C API | Description |
|---|---|
| All `OrtEpApi` functions | EP helper functions (CreateEpDevice, KernelRegistry, KernelDefBuilder, etc.) |
| `OrtEp`, `OrtEpFactory` structs | EP-implemented callback interfaces |
| `OrtDataTransferImpl`, `OrtSyncStreamImpl`, `OrtSyncNotificationImpl` | EP data transfer/sync primitives |
| `OrtKernelImpl`, `OrtNodeComputeInfo` | EP kernel implementation |
| `OrtExternalResourceImporterImpl` | EP external resource import |
| `GetExecutionProviderApi`, `GetEpApi` | Get EP-specific API struct pointer |

### Custom Allocator / Runtime Hosting

| C API | Description |
|---|---|
| `CreateAndRegisterAllocator`, `RegisterAllocator`, `UnregisterAllocator`, `CreateAndRegisterAllocatorV2` | Custom allocator registration |
| `CreateEnvWithCustomLogger`, `CreateEnvWithCustomLoggerAndGlobalThreadPools` | Custom logger env creation |
| `SetLanguageProjection` | Set language projection |
| All `SessionOptionsSetCustom*ThreadFn`, `SetGlobal*ThreadFn` | Custom threading |
| `CreateThreadingOptions` + all `SetGlobal*` threading APIs | Threading options management |
| `CreateArenaCfg`, `CreateArenaCfgV2` | Arena configuration |
| `CreatePrepackedWeightsContainer` + prepacked session creation | Advanced weight sharing |

### Graph Inspection / Model Editing

| C API | Description |
|---|---|
| All `Graph_*` (~20 functions) | Graph name, path, IR version, opsets, inputs, outputs, initializers, nodes |
| All `Node_*` (~15 functions) | Node id, name, op type, domain, inputs, outputs, attributes, subgraphs, EP name |
| All `ValueInfo_*` (~10 functions) | Value producer/consumers, initializer info, graph input/output checks |
| All `OpAttr_*` | Attribute name, type, tensor value |
| All `ExternalInitializerInfo_*` | File path, offset, byte size |
| All `EpAssignedSubgraph_*`, `EpAssignedNode_*` | EP assignment inspection |
| `GetModelEditorApi` | Model editor API |

### GPU Interop (InteropApi)

| C API | Description |
|---|---|
| `CreateExternalResourceImporterForDevice` | Create external resource importer |
| `CanImportMemory`, `ImportMemory`, `ReleaseExternalMemoryHandle` | Import external GPU memory |
| `CreateTensorFromMemory` | Create tensor from imported memory |
| `CanImportSemaphore`, `ImportSemaphore`, `ReleaseExternalSemaphoreHandle` | Import GPU semaphores |
| `WaitSemaphore`, `SignalSemaphore` | Semaphore synchronization |

### Sparse Tensor (Niche)

| C API | Description |
|---|---|
| `IsSparseTensor` | Check if value is sparse |
| `CreateSparseTensorAsOrtValue`, `CreateSparseTensorWithValuesAsOrtValue` | Create sparse tensors |
| `FillSparseTensorCoo`, `FillSparseTensorCsr`, `FillSparseTensorBlockSparse` | Fill sparse data |
| `UseCooIndices`, `UseCsrIndices`, `UseBlockSparseIndices` | Set sparse indices |
| `GetSparseTensorFormat`, `GetSparseTensorValuesTypeAndShape`, `GetSparseTensorValues` | Read sparse data |
| `GetSparseTensorIndicesTypeShape`, `GetSparseTensorIndices` | Read sparse indices |

### Opaque Value / TypeInfo Creation

| C API | Description |
|---|---|
| `CreateOpaqueValue`, `GetOpaqueValue` | Custom type marshaling |
| `CreateTensorTypeAndShapeInfo`, `SetTensorElementType`, `SetDimensions`, `SetSymbolicDimensions` | Type manipulation |
| `SetCurrentGpuDeviceId`, `GetCurrentGpuDeviceId` | Low-level GPU device management |

---

## Summary

| Priority | Group | API Count | Use Case |
|---|---|---|---|
| **Medium** | Group 1 — Value/Tensor extended | 15 | String tensors, map/sequence values (major design change needed) |
| **Medium** | Group 2 — TypeInfo Map/Seq/Optional | 7 | Map/sequence/optional type introspection |
| N/A | Training APIs | ~40+ | **Not inference** |
| N/A | Custom Op / KernelInfo / KernelContext | ~40+ | **Not for normal Python use** |
| N/A | EP implementation | ~60+ | **Not for normal Python use** |
| N/A | Graph inspection | ~50 | **Not for normal Python use** |
| N/A | InteropApi | ~11 | **Not for normal Python use** |
| N/A | Sparse Tensor | ~14 | **Niche** |
| N/A | Runtime hosting / Allocator registration | ~15 | **Not for normal Python use** |
| N/A | Env extended | 3 | **Not supported** |
| N/A | Overridable Initializers | 3 | **Niche — not supported** |
| N/A | Initializer Management | 3 | **Deferred — needs OrtValue initializer support** |
| N/A | Callback-based APIs | 4 | **Requires C callback bridging** |
| N/A | Async execution | 1 | **Requires C callback bridging** |
| N/A | SyncStream / Raw pointers / Allocator | ~15 | **Not for normal Python use** |
| N/A | Legacy per-EP registration | ~30 | **V1 struct APIs, V2 Create/Update/Release, serialization helpers — covered by generic `append_execution_provider`** |

# Missing ORT C API Bindings

> **Minimum target: ORT 1.23 (API version 23).** All currently bound APIs are v22 or earlier — no version guards needed.
> APIs marked with a version require `#if ORT_API_VERSION >= N` guards when implemented.

## Currently Bound APIs (~80 C API calls)

| Area | What's Covered |
|---|---|
| **Env** | `CreateEnv`, `ReleaseEnv`, `DisableTelemetryEvents`, `RegisterExecutionProviderLibrary`, `UnregisterExecutionProviderLibrary`, `GetEpDevices`, `UpdateEnvWithCustomLogLevel` |
| **SessionOptions** | Create/Release, `SetOptimizedModelFilePath`, `SetSessionExecutionMode`, Enable/Disable Profiling, Enable/Disable MemPattern, Enable/Disable CpuMemArena, `SetSessionLogId`, `SetSessionLogVerbosityLevel`, `SetSessionLogSeverityLevel`, `SetSessionGraphOptimizationLevel`, `SetIntraOpNumThreads`, `SetInterOpNumThreads`, `RegisterCustomOpsLibrary` (V1), `SessionOptionsAppendExecutionProvider_V2`, `SessionOptionsSetEpSelectionPolicy`, `SessionOptionsSetEpSelectionPolicyDelegate` |
| **Session** | `CreateSession`, `CreateSessionFromArray`, `Run`, `RunWithBinding`, `SessionGetInputCount/Name/TypeInfo`, `SessionGetOutputCount/Name/TypeInfo`, `SessionGetModelMetadata`, `SessionEndProfiling`, `SessionGetProfilingStartTimeNs`, `SessionGetMemoryInfoForInputs/Outputs`, `SessionGetEpDeviceForInputs/Outputs` (v24), `Session_GetEpGraphAssignmentInfo` (v24) |
| **RunOptions** | Create/Release, Set/Get RunLogVerbosityLevel, Set/Get RunLogSeverityLevel, Set/Get RunTag, Set/Unset Terminate |
| **Value/Tensor** | `CreateTensorAsOrtValue`, `CreateTensorWithDataAsOrtValue`, `GetTensorMutableData`, `GetTensorTypeAndShape`, `GetTensorElementType`, `GetDimensionsCount`, `GetDimensions`, `GetSymbolicDimensions`, `ReleaseValue` |
| **TypeInfo** | `CastTypeInfoToTensorInfo`, `ReleaseTypeInfo`, `ReleaseTensorTypeAndShapeInfo` |
| **MemoryInfo** | `CreateCpuMemoryInfo`, `CreateMemoryInfo`, `MemoryInfoGetName`, `MemoryInfoGetId`, `MemoryInfoGetMemType`, `MemoryInfoGetType`, `MemoryInfoGetDeviceType`, `CompareMemoryInfo`, `ReleaseMemoryInfo` |
| **Allocator** | `GetAllocatorWithDefaultOptions` |
| **EpDevice/HardwareDevice** | `HardwareDevice_Type/VendorId/Vendor/DeviceId/Metadata`, `EpDevice_EpName/EpVendor/EpMetadata/EpOptions/Device` |
| **ModelMetadata** | `ReleaseModelMetadata`, `ModelMetadataGetProducerName/GraphName/Domain/Description/GraphDescription/Version`, `ModelMetadataLookupCustomMetadataMap`, `ModelMetadataGetCustomMetadataMapKeys` |
| **IoBinding** | `CreateIoBinding`, `ReleaseIoBinding`, `BindInput`, `BindOutput`, `BindOutputToDevice`, `GetBoundOutputNames`, `GetBoundOutputValues`, `ClearBoundInputs`, `ClearBoundOutputs`, `SynchronizeBoundInputs`, `SynchronizeBoundOutputs` |
| **CompileApi** | `CreateModelCompilationOptionsFromSessionOptions`, `ModelCompilationOptions_SetInputModelPath`, `SetInputModelFromBuffer`, `SetOutputModelExternalInitializersFile`, `SetEpContextEmbedMode`, `SetOutputModelPath`, `SetOutputModelBuffer`, `CompileModel`, `ReleaseModelCompilationOptions` |

---

## Missing APIs — Inference-Related, Normal Python Use

### Group 1: SessionOptions — Additional Configuration (Medium Priority)

Commonly needed for advanced session setup.

| C API | Description | Status | API Ver |
|---|---|---|---|
| `CloneSessionOptions` | Clone session options | | ≤10 |
| `AddFreeDimensionOverride` | Override free dim by denotation | | ≤10 |
| `AddFreeDimensionOverrideByName` | Override free dim by name | | ≤10 |
| `DisablePerSessionThreads` | Use global thread pools instead | | ≤10 |
| `AddInitializer` | Add pre-loaded initializer | | ≤10 |
| `AddExternalInitializers` | Replace external initializers with tensors | | 12 |
| `AddExternalInitializersFromFilesInMemory` | Add external initializers from memory buffers | | 18 |
| `AddSessionConfigEntry` | Add a config key-value pair | | ≤10 |
| `HasSessionConfigEntry` | Check if a config key exists | | 14 |
| `GetSessionConfigEntry` | Get a config value by key | | 14 |
| `GetSessionOptionsConfigEntries` | Get all config entries | | 23 |
| `RegisterCustomOpsLibrary_V2` | Register custom ops (V2, replaces deprecated V1) | | 14 |
| `RegisterCustomOpsUsingFunction` | Register custom ops via named function | | 14 |
| `EnableOrtCustomOps` | Enable built-in ORT custom ops | | ≤10 |
| `SetUserLoggingFunction` | Set custom user logging function | | 17 |
| `SetDeterministicCompute` | Force deterministic GPU compute | | 17 |
| `SessionOptionsSetLoadCancellationFlag` | Set flag to cancel session loading | | 22 |

### Group 2: RunOptions — Additional Configuration (Medium Priority)

| C API | Description | Status | API Ver |
|---|---|---|---|
| `AddRunConfigEntry` | Add config key-value to run options | | ≤10 |
| `GetRunConfigEntry` | Get config entry from run options | | 23 |
| `RunOptionsSetSyncStream` | Set `OrtSyncStream` on run options | | **24** |

### Group 3: Value/Tensor — Extended Operations (Medium Priority)

String tensor support, additional tensor queries, and non-tensor value support (maps, sequences — rarely used in practice).

| C API | Description | Status | API Ver |
|---|---|---|---|
| `IsTensor` | Check if an OrtValue is a tensor | | ≤10 |
| `GetValueType` | Get ONNXType from OrtValue | | ≤10 |
| `HasValue` | Check if optional OrtValue has data | | ≤10 |
| `GetTensorMemoryInfo` | Get memory info of a tensor | | ≤10 |
| `GetTensorSizeInBytes` | Get total tensor size in bytes | | 23 |
| `GetTensorData` | Get const pointer to tensor data | | 23 |
| `FillStringTensor` | Fill a string tensor | | ≤10 |
| `GetStringTensorDataLength` | Get total string tensor byte length | | ≤10 |
| `GetStringTensorContent` | Get all string tensor content | | ≤10 |
| `GetStringTensorElementLength` | Get length of one string element | | ≤10 |
| `GetStringTensorElement` | Get one string element | | ≤10 |
| `FillStringTensorElement` | Fill one string element | | ≤10 |
| `GetResizedStringTensorElementBuffer` | Get resized buffer for a string element | | 15 |
| `TensorAt` | Get pointer to a specific tensor element | | ≤10 |
| `GetValue` | Extract element from map/sequence OrtValue | | ≤10 |
| `GetValueCount` | Get count in map/sequence OrtValue | | ≤10 |
| `CreateValue` | Create a map/sequence OrtValue | | ≤10 |
| `CreateTensorWithDataAndDeleterAsOrtValue` | Create tensor with custom deleter | | 22 |

### Group 4: Async Execution (Low Priority)

| C API | Description | Status | API Ver |
|---|---|---|---|
| `RunAsync` | Run model asynchronously | | 16 |

### Group 5: TypeInfo — Extended (Medium Priority)

Maps, sequences, optionals type introspection.

| C API | Description | Status | API Ver |
|---|---|---|---|
| `GetOnnxTypeFromTypeInfo` | Get ONNXType from TypeInfo | | ≤10 |
| `GetDenotationFromTypeInfo` | Get denotation string | | ≤10 |
| `CastTypeInfoToMapTypeInfo` | Cast to OrtMapTypeInfo | | ≤10 |
| `CastTypeInfoToSequenceTypeInfo` | Cast to OrtSequenceTypeInfo | | ≤10 |
| `CastTypeInfoToOptionalTypeInfo` | Cast to OrtOptionalTypeInfo | | 15 |
| `GetMapKeyType` | Get map key element type | | ≤10 |
| `GetMapValueType` | Get map value type info | | ≤10 |
| `GetSequenceElementType` | Get sequence element type info | | ≤10 |
| `GetOptionalContainedTypeInfo` | Get contained type of optional | | 15 |
| `GetTensorShapeElementCount` | Get total element count | | ≤10 |

### Group 6: Allocator — Extended (Low Priority)

| C API | Description | Status | API Ver |
|---|---|---|---|
| `CreateAllocator` | Create allocator from session + MemoryInfo | | ≤10 |
| `AllocatorAlloc` | Allocate memory | | ≤10 |
| `AllocatorFree` | Free memory | | ≤10 |
| `AllocatorGetInfo` | Get allocator's MemoryInfo | | ≤10 |

### Group 7: Legacy Per-EP Registration (Low Priority)

> The project already uses the V2 EP API (`SessionOptionsAppendExecutionProvider_V2` + `GetEpDevices`). These are the older per-EP APIs. Whether to bind them depends on backward compatibility goals.

| C API | Description | Status | API Ver |
|---|---|---|---|
| `SessionOptionsAppendExecutionProvider_CUDA` | Append CUDA EP (V1) | | ≤10 |
| `SessionOptionsAppendExecutionProvider_CUDA_V2` | Append CUDA EP (V2) | | ≤10 |
| `SessionOptionsAppendExecutionProvider_ROCM` | Append ROCm EP | | ≤10 |
| `SessionOptionsAppendExecutionProvider_OpenVINO` | Append OpenVINO EP (V1) | | ≤10 |
| `SessionOptionsAppendExecutionProvider_OpenVINO_V2` | Append OpenVINO EP (V2) | | 17 |
| `SessionOptionsAppendExecutionProvider_TensorRT` | Append TensorRT EP (V1) | | ≤10 |
| `SessionOptionsAppendExecutionProvider_TensorRT_V2` | Append TensorRT EP (V2) | | ≤10 |
| `SessionOptionsAppendExecutionProvider_MIGraphX` | Append MIGraphX EP | | ≤10 |
| `SessionOptionsAppendExecutionProvider_CANN` | Append CANN EP | | 14 |
| `SessionOptionsAppendExecutionProvider_Dnnl` | Append oneDNN/DNNL EP | | 14 |
| `SessionOptionsAppendExecutionProvider` | Generic EP append with key-value options | | 12 |
| `SessionOptionsAppendExecutionProvider_VitisAI` | Append VitisAI EP | | 17 |
| `GetAvailableProviders` | Get list of available EP names | | ≤10 |
| `SetEpDynamicOptions` | Set dynamic EP options at runtime | | 20 |
| All `Create/Update/Get/Release*ProviderOptions` | Provider-specific option management | | varies |

### Group 8: SyncStream / Data Transfer / Shared Allocator (Low Priority)

| C API | Description | Status | API Ver |
|---|---|---|---|
| `CreateSyncStreamForEpDevice` | Create sync stream for an EP device | | 23 |
| `SyncStream_GetHandle` | Get native stream handle | | 23 |
| `CopyTensors` | Copy tensors between devices | | 23 |
| `CreateSharedAllocator` | Create shared allocator for EP device | | 23 |
| `GetSharedAllocator` | Get shared allocator from env | | 23 |
| `ReleaseSharedAllocator` | Release shared allocator | | 23 |

### Group 9: EpDevice / HardwareDevice — Extended Query (Medium Priority)

| C API | Description | Status | API Ver |
|---|---|---|---|
| `EpDevice_MemoryInfo` | Get MemoryInfo for an EpDevice | | 23 |
| `GetNumHardwareDevices` | Get number of hardware devices | | **24** |
| `GetHardwareDevices` | Get all hardware devices | | **24** |
| `GetHardwareDeviceEpIncompatibilityDetails` | Check HW/EP incompatibility | | **24** |
| `GetModelCompatibilityForEpDevices` | Validate compiled model compatibility | | 23 |
| `GetCompatibilityInfoFromModel` | Check compatibility from model file | | **24** |
| `GetCompatibilityInfoFromModelBytes` | Check compatibility from model bytes | | **24** |

### Group 10: LoRA Adapter (Medium Priority)

| C API | Description | Status | API Ver |
|---|---|---|---|
| `CreateLoraAdapter` | Create LoRA adapter from file | | 20 |
| `CreateLoraAdapterFromArray` | Create LoRA adapter from byte array | | 20 |
| `RunOptionsAddActiveLoraAdapter` | Activate LoRA adapter for a run | | 20 |

### Group 11: CompileApi — Additional Options (Medium Priority)

| C API | Description | Status | API Ver |
|---|---|---|---|
| `ModelCompilationOptions_SetFlags` | Set boolean compilation flags | | 23 (CompileApi) |
| `ModelCompilationOptions_SetEpContextBinaryInformation` | Set EP context binary info | | 23 (CompileApi) |
| `ModelCompilationOptions_SetGraphOptimizationLevel` | Set graph opt level for compilation | | 23 (CompileApi) |
| `ModelCompilationOptions_SetOutputModelWriteFunc` | Set custom write function | | 23 (CompileApi) |
| `ModelCompilationOptions_SetOutputModelGetInitializerLocationFunc` | Set initializer location function | | 23 (CompileApi) |

---

## Missing APIs — NOT Inference-Related

### Training APIs

| C API | Description |
|---|---|
| `GetTrainingApi` + all `OrtTrainingApi` functions | Full training API surface (defined in `onnxruntime_training_c_api.h`) |

---

## Missing APIs — NOT for Normal Python Use

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
| **Medium** | Group 3 — Value/Tensor extended | 18 | String tensors, map/sequence values (non-tensor values are rare) |
| **Medium** | Group 1 — SessionOptions extended | 17 | Free dim overrides, config entries, custom ops V2 |
| **Medium** | Group 2 — RunOptions extended | 3 | Config entries, sync stream |
| **Medium** | Group 5 — TypeInfo extended | 10 | Map/sequence/optional type introspection |
| **Medium** | Group 9 — EpDevice/HW extended | 7 | Compatibility checks, hardware enumeration |
| **Medium** | Group 10 — LoRA Adapter | 3 | LoRA fine-tuned model inference |
| **Medium** | Group 11 — CompileApi extended | 5 | Advanced compilation options |
| **Low** | Group 4 — Async execution | 1 | Async inference |
| **Low** | Group 6 — Allocator extended | 4 | Custom allocation |
| **Low** | Group 7 — Legacy per-EP registration | ~25 | Old-style EP setup (project uses V2) |
| **Low** | Group 8 — SyncStream/transfer | 6 | Cross-device tensor copies |
| N/A | Training APIs | ~40+ | **Not inference** |
| N/A | Custom Op / KernelInfo / KernelContext | ~40+ | **Not for normal Python use** |
| N/A | EP implementation | ~60+ | **Not for normal Python use** |
| N/A | Graph inspection | ~50 | **Not for normal Python use** |
| N/A | InteropApi | ~11 | **Not for normal Python use** |
| N/A | Sparse Tensor | ~14 | **Niche** |
| N/A | Runtime hosting / Allocator registration | ~15 | **Not for normal Python use** |
| N/A | Env extended | 3 | **Not supported** |
| N/A | Overridable Initializers | 3 | **Niche — not supported** |

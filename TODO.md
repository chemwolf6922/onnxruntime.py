# Missing ORT C API Bindings

## Currently Bound APIs (~80 C API calls)

| Area | What's Covered |
|---|---|
| **Env** | `CreateEnv`, `ReleaseEnv`, `DisableTelemetryEvents`, `RegisterExecutionProviderLibrary`, `UnregisterExecutionProviderLibrary`, `GetEpDevices` |
| **SessionOptions** | Create/Release, `SetOptimizedModelFilePath`, `SetSessionExecutionMode`, Enable/Disable Profiling, Enable/Disable MemPattern, Enable/Disable CpuMemArena, `SetSessionLogId`, `SetSessionLogVerbosityLevel`, `SetSessionLogSeverityLevel`, `SetSessionGraphOptimizationLevel`, `SetIntraOpNumThreads`, `SetInterOpNumThreads`, `RegisterCustomOpsLibrary` (V1), `SessionOptionsAppendExecutionProvider_V2`, `SessionOptionsSetEpSelectionPolicy`, `SessionOptionsSetEpSelectionPolicyDelegate` |
| **Session** | `CreateSession`, `CreateSessionFromArray`, `Run`, `RunWithBinding`, `SessionGetInputCount/Name/TypeInfo`, `SessionGetOutputCount/Name/TypeInfo`, `SessionGetModelMetadata`, `SessionEndProfiling`, `SessionGetProfilingStartTimeNs` |
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

### Group 1: Session — Additional Queries (Medium Priority)

Useful for introspection of loaded sessions.

| C API | Description | Status |
|---|---|---|
| `SessionGetOverridableInitializerCount` | Count of overridable initializers | |
| `SessionGetOverridableInitializerName` | Name of an overridable initializer | |
| `SessionGetOverridableInitializerTypeInfo` | TypeInfo of overridable initializer | |
| `SessionGetMemoryInfoForInputs` | Get `OrtMemoryInfo` for each input | |
| `SessionGetMemoryInfoForOutputs` | Get `OrtMemoryInfo` for each output | |
| `SessionGetEpDeviceForInputs` | Get `OrtEpDevice` assigned per input | |
| `SessionGetEpDeviceForOutputs` | Get `OrtEpDevice` assigned per output | |
| `Session_GetEpGraphAssignmentInfo` | Get EP graph assignment info | |

### Group 2: SessionOptions — Additional Configuration (Medium Priority)

Commonly needed for advanced session setup.

| C API | Description | Status |
|---|---|---|
| `CloneSessionOptions` | Clone session options | |
| `AddFreeDimensionOverride` | Override free dim by denotation | |
| `AddFreeDimensionOverrideByName` | Override free dim by name | |
| `DisablePerSessionThreads` | Use global thread pools instead | |
| `AddInitializer` | Add pre-loaded initializer | |
| `AddExternalInitializers` | Replace external initializers with tensors | |
| `AddExternalInitializersFromFilesInMemory` | Add external initializers from memory buffers | |
| `AddSessionConfigEntry` | Add a config key-value pair | |
| `HasSessionConfigEntry` | Check if a config key exists | |
| `GetSessionConfigEntry` | Get a config value by key | |
| `GetSessionOptionsConfigEntries` | Get all config entries | |
| `RegisterCustomOpsLibrary_V2` | Register custom ops (V2, replaces deprecated V1) | |
| `RegisterCustomOpsUsingFunction` | Register custom ops via named function | |
| `EnableOrtCustomOps` | Enable built-in ORT custom ops | |
| `SetUserLoggingFunction` | Set custom user logging function | |
| `SetDeterministicCompute` | Force deterministic GPU compute | |
| `SessionOptionsSetLoadCancellationFlag` | Set flag to cancel session loading | |

### Group 3: RunOptions — Additional Configuration (Medium Priority)

| C API | Description | Status |
|---|---|---|
| `AddRunConfigEntry` | Add config key-value to run options | |
| `GetRunConfigEntry` | Get config entry from run options | |
| `RunOptionsSetSyncStream` | Set `OrtSyncStream` on run options | |

### Group 4: Value/Tensor — Extended Operations (Medium Priority)

String tensor support, additional tensor queries, and non-tensor value support (maps, sequences — rarely used in practice).

| C API | Description | Status |
|---|---|---|
| `IsTensor` | Check if an OrtValue is a tensor | |
| `GetValueType` | Get ONNXType from OrtValue | |
| `HasValue` | Check if optional OrtValue has data | |
| `GetTensorMemoryInfo` | Get memory info of a tensor | |
| `GetTensorSizeInBytes` | Get total tensor size in bytes | |
| `GetTensorData` | Get const pointer to tensor data | |
| `FillStringTensor` | Fill a string tensor | |
| `GetStringTensorDataLength` | Get total string tensor byte length | |
| `GetStringTensorContent` | Get all string tensor content | |
| `GetStringTensorElementLength` | Get length of one string element | |
| `GetStringTensorElement` | Get one string element | |
| `FillStringTensorElement` | Fill one string element | |
| `GetResizedStringTensorElementBuffer` | Get resized buffer for a string element | |
| `TensorAt` | Get pointer to a specific tensor element | |
| `GetValue` | Extract element from map/sequence OrtValue | |
| `GetValueCount` | Get count in map/sequence OrtValue | |
| `CreateValue` | Create a map/sequence OrtValue | |
| `CreateTensorWithDataAndDeleterAsOrtValue` | Create tensor with custom deleter | |

### Group 5: Async Execution (Low Priority)

| C API | Description | Status |
|---|---|---|
| `RunAsync` | Run model asynchronously | |

### Group 6: TypeInfo — Extended (Medium Priority)

Maps, sequences, optionals type introspection.

| C API | Description | Status |
|---|---|---|
| `GetOnnxTypeFromTypeInfo` | Get ONNXType from TypeInfo | |
| `GetDenotationFromTypeInfo` | Get denotation string | |
| `CastTypeInfoToMapTypeInfo` | Cast to OrtMapTypeInfo | |
| `CastTypeInfoToSequenceTypeInfo` | Cast to OrtSequenceTypeInfo | |
| `CastTypeInfoToOptionalTypeInfo` | Cast to OrtOptionalTypeInfo | |
| `GetMapKeyType` | Get map key element type | |
| `GetMapValueType` | Get map value type info | |
| `GetSequenceElementType` | Get sequence element type info | |
| `GetOptionalContainedTypeInfo` | Get contained type of optional | |
| `GetTensorShapeElementCount` | Get total element count | |

### Group 7: Env — Extended Configuration (Medium Priority)

| C API | Description | Status |
|---|---|---|
| `CreateEnvWithGlobalThreadPools` | Create env with global thread pools | |
| `EnableTelemetryEvents` | Enable telemetry | |
| `UpdateEnvWithCustomLogLevel` | Update log severity level | |
| `GetBuildInfoString` | Get ORT build info string | |

### Group 8: Allocator — Extended (Low Priority)

| C API | Description | Status |
|---|---|---|
| `CreateAllocator` | Create allocator from session + MemoryInfo | |
| `AllocatorAlloc` | Allocate memory | |
| `AllocatorFree` | Free memory | |
| `AllocatorGetInfo` | Get allocator's MemoryInfo | |

### Group 9: Legacy Per-EP Registration (Low Priority)

> The project already uses the V2 EP API (`SessionOptionsAppendExecutionProvider_V2` + `GetEpDevices`). These are the older per-EP APIs. Whether to bind them depends on backward compatibility goals.

| C API | Description | Status |
|---|---|---|
| `SessionOptionsAppendExecutionProvider_CUDA` | Append CUDA EP (V1) | |
| `SessionOptionsAppendExecutionProvider_CUDA_V2` | Append CUDA EP (V2) | |
| `SessionOptionsAppendExecutionProvider_ROCM` | Append ROCm EP | |
| `SessionOptionsAppendExecutionProvider_OpenVINO` | Append OpenVINO EP (V1) | |
| `SessionOptionsAppendExecutionProvider_OpenVINO_V2` | Append OpenVINO EP (V2) | |
| `SessionOptionsAppendExecutionProvider_TensorRT` | Append TensorRT EP (V1) | |
| `SessionOptionsAppendExecutionProvider_TensorRT_V2` | Append TensorRT EP (V2) | |
| `SessionOptionsAppendExecutionProvider_MIGraphX` | Append MIGraphX EP | |
| `SessionOptionsAppendExecutionProvider_CANN` | Append CANN EP | |
| `SessionOptionsAppendExecutionProvider_Dnnl` | Append oneDNN/DNNL EP | |
| `SessionOptionsAppendExecutionProvider` | Generic EP append with key-value options | |
| `SessionOptionsAppendExecutionProvider_VitisAI` | Append VitisAI EP | |
| `GetAvailableProviders` | Get list of available EP names | |
| `SetEpDynamicOptions` | Set dynamic EP options at runtime | |
| All `Create/Update/Get/Release*ProviderOptions` | Provider-specific option management | |

### Group 10: SyncStream / Data Transfer / Shared Allocator (Low Priority)

| C API | Description | Status |
|---|---|---|
| `CreateSyncStreamForEpDevice` | Create sync stream for an EP device | |
| `SyncStream_GetHandle` | Get native stream handle | |
| `CopyTensors` | Copy tensors between devices | |
| `CreateSharedAllocator` | Create shared allocator for EP device | |
| `GetSharedAllocator` | Get shared allocator from env | |
| `ReleaseSharedAllocator` | Release shared allocator | |

### Group 11: EpDevice / HardwareDevice — Extended Query (Medium Priority)

| C API | Description | Status |
|---|---|---|
| `EpDevice_MemoryInfo` | Get MemoryInfo for an EpDevice | |
| `GetNumHardwareDevices` | Get number of hardware devices | |
| `GetHardwareDevices` | Get all hardware devices | |
| `GetHardwareDeviceEpIncompatibilityDetails` | Check HW/EP incompatibility | |
| `GetModelCompatibilityForEpDevices` | Validate compiled model compatibility | |
| `GetCompatibilityInfoFromModel` | Check compatibility from model file | |
| `GetCompatibilityInfoFromModelBytes` | Check compatibility from model bytes | |

### Group 12: LoRA Adapter (Medium Priority)

| C API | Description | Status |
|---|---|---|
| `CreateLoraAdapter` | Create LoRA adapter from file | |
| `CreateLoraAdapterFromArray` | Create LoRA adapter from byte array | |
| `RunOptionsAddActiveLoraAdapter` | Activate LoRA adapter for a run | |

### Group 13: CompileApi — Additional Options (Medium Priority)

| C API | Description | Status |
|---|---|---|
| `ModelCompilationOptions_SetFlags` | Set boolean compilation flags | |
| `ModelCompilationOptions_SetEpContextBinaryInformation` | Set EP context binary info | |
| `ModelCompilationOptions_SetGraphOptimizationLevel` | Set graph opt level for compilation | |
| `ModelCompilationOptions_SetOutputModelWriteFunc` | Set custom write function | |
| `ModelCompilationOptions_SetOutputModelGetInitializerLocationFunc` | Set initializer location function | |

---

## Missing APIs — NOT Inference-Related

### Training APIs

| C API | Description |
|---|---|
| `GetTrainingApi` + all `OrtTrainingApi` functions | Full training API surface (defined in `onnxruntime_training_c_api.h`) |

---

## Missing APIs — NOT for Normal Python Use

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
| **Medium** | Group 4 — Value/Tensor extended | 18 | String tensors, map/sequence values (non-tensor values are rare) |
| **Medium** | Group 2 — SessionOptions extended | 17 | Free dim overrides, config entries, custom ops V2 |
| **Medium** | Group 1 — Session queries | 8 | Per-input memory/device info, EP assignment |
| **Medium** | Group 3 — RunOptions extended | 3 | Config entries, sync stream |
| **Medium** | Group 6 — TypeInfo extended | 10 | Map/sequence/optional type introspection |
| **Medium** | Group 7 — Env extended | 4 | Global thread pools, build info |
| **Medium** | Group 11 — EpDevice/HW extended | 7 | Compatibility checks, hardware enumeration |
| **Medium** | Group 12 — LoRA Adapter | 3 | LoRA fine-tuned model inference |
| **Medium** | Group 13 — CompileApi extended | 5 | Advanced compilation options |
| **Low** | Group 5 — Async execution | 1 | Async inference |
| **Low** | Group 8 — Allocator extended | 4 | Custom allocation |
| **Low** | Group 9 — Legacy per-EP registration | ~25 | Old-style EP setup (project uses V2) |
| **Low** | Group 10 — SyncStream/transfer | 6 | Cross-device tensor copies |
| N/A | Training APIs | ~40+ | **Not inference** |
| N/A | Custom Op / KernelInfo / KernelContext | ~40+ | **Not for normal Python use** |
| N/A | EP implementation | ~60+ | **Not for normal Python use** |
| N/A | Graph inspection | ~50 | **Not for normal Python use** |
| N/A | InteropApi | ~11 | **Not for normal Python use** |
| N/A | Sparse Tensor | ~14 | **Niche** |
| N/A | Runtime hosting / Allocator registration | ~15 | **Not for normal Python use** |

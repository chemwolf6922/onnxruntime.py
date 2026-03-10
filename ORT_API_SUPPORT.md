# ORT C API Support Matrix

This document maps every function pointer in the ORT C API structs (`OrtApi`, `OrtCompileApi`, `OrtInteropApi`)
to whether it is supported in the `ortpy` binding, following the declaration order in
`onnxruntime_c_api.h`.

Legend:
- ✅ Supported — the API is exposed (directly or indirectly) in the Python binding.
- ❌ Not supported — the API is not currently exposed.
- ➖ Out of scope — the API is intentionally not exposed and there are no plans to support it.

---

## OrtApi — Base (≤ 1.10)

| | API | Detail |
|---|---|---|
| ✅ | CreateStatus | `Status` class (internal) |
| ✅ | GetErrorCode | `Status.GetErrorCode()` (internal) |
| ✅ | GetErrorMessage | `Status.GetErrorMessage()` (internal) |
| ✅ | CreateEnv | `Env` singleton (auto-created) |
| ✅ | CreateEnvWithCustomLogger | `ortpy.create_env(logging_function=...)` |
| ➖ | EnableTelemetryEvents | Telemetry is always disabled |
| ➖ | DisableTelemetryEvents | Telemetry is always disabled |
| ✅ | CreateSession | `Session(model_path, options)` |
| ✅ | CreateSessionFromArray | `Session(model_bytes, options)` |
| ✅ | Run | `Session.run()` / `Session.run_with_ort_values()` |
| ✅ | CreateSessionOptions | `SessionOptions()` |
| ✅ | SetOptimizedModelFilePath | `SessionOptions.set_optimized_model_file_path()` |
| ✅ | CloneSessionOptions | `SessionOptions.clone()` |
| ✅ | SetSessionExecutionMode | `SessionOptions.set_session_execution_mode()` |
| ✅ | EnableProfiling | `SessionOptions.enable_profiling()` |
| ✅ | DisableProfiling | `SessionOptions.disable_profiling()` |
| ✅ | EnableMemPattern | `SessionOptions.enable_mem_pattern()` |
| ✅ | DisableMemPattern | `SessionOptions.disable_mem_pattern()` |
| ✅ | EnableCpuMemArena | `SessionOptions.enable_cpu_mem_arena()` |
| ✅ | DisableCpuMemArena | `SessionOptions.disable_cpu_mem_arena()` |
| ✅ | SetSessionLogId | `SessionOptions.set_session_log_id()` |
| ✅ | SetSessionLogVerbosityLevel | `SessionOptions.set_session_log_verbosity_level()` |
| ✅ | SetSessionLogSeverityLevel | `SessionOptions.set_session_log_severity_level()` |
| ✅ | SetSessionGraphOptimizationLevel | `SessionOptions.set_session_graph_optimization_level()` |
| ✅ | SetIntraOpNumThreads | `SessionOptions.set_intra_op_num_threads()` |
| ✅ | SetInterOpNumThreads | `SessionOptions.set_inter_op_num_threads()` |
| ➖ | CreateCustomOpDomain | Custom op implementation not in scope |
| ➖ | CustomOpDomain_Add | Custom op implementation not in scope |
| ➖ | AddCustomOpDomain | Custom op implementation not in scope |
| ✅ | RegisterCustomOpsLibrary | `SessionOptions.register_custom_ops_library()` |
| ✅ | SessionGetInputCount | Used internally by `Session.get_input_info()` |
| ✅ | SessionGetOutputCount | Used internally by `Session.get_output_info()` |
| ✅ | SessionGetOverridableInitializerCount | Used internally by `Session.get_overridable_initializer_info()` |
| ✅ | SessionGetInputTypeInfo | Used internally by `Session.get_input_info()` |
| ✅ | SessionGetOutputTypeInfo | Used internally by `Session.get_output_info()` |
| ✅ | SessionGetOverridableInitializerTypeInfo | Used internally by `Session.get_overridable_initializer_info()` |
| ✅ | SessionGetInputName | Used internally by `Session.get_input_info()` |
| ✅ | SessionGetOutputName | Used internally by `Session.get_output_info()` |
| ✅ | SessionGetOverridableInitializerName | Used internally by `Session.get_overridable_initializer_info()` |
| ✅ | CreateRunOptions | `RunOptions()` |
| ✅ | RunOptionsSetRunLogVerbosityLevel | `RunOptions.run_log_verbosity_level` (setter) |
| ✅ | RunOptionsSetRunLogSeverityLevel | `RunOptions.run_log_severity_level` (setter) |
| ✅ | RunOptionsSetRunTag | `RunOptions.run_tag` (setter) |
| ✅ | RunOptionsGetRunLogVerbosityLevel | `RunOptions.run_log_verbosity_level` (getter) |
| ✅ | RunOptionsGetRunLogSeverityLevel | `RunOptions.run_log_severity_level` (getter) |
| ✅ | RunOptionsGetRunTag | `RunOptions.run_tag` (getter) |
| ✅ | RunOptionsSetTerminate | `RunOptions.set_terminate()` |
| ✅ | RunOptionsUnsetTerminate | `RunOptions.unset_terminate()` |
| ✅ | CreateTensorAsOrtValue | `Value(shape, type)` constructor (internal) |
| ✅ | CreateTensorWithDataAsOrtValue | `Value(numpy_array)` constructor |
| ✅ | IsTensor | `Value.is_tensor` |
| ✅ | GetTensorMutableData | Used internally by `Value.numpy()` |
| ✅ | FillStringTensor | Used internally by `Value.from_strings()` |
| ✅ | GetStringTensorDataLength | Used internally by `Value.get_strings()` |
| ✅ | GetStringTensorContent | Used internally by `Value.get_strings()` |
| ✅ | CastTypeInfoToTensorInfo | Used internally by `TypeInfo` tensor accessors |
| ✅ | GetOnnxTypeFromTypeInfo | `TypeInfo.onnx_type` |
| ➖ | CreateTensorTypeAndShapeInfo | Standalone creation not exposed |
| ➖ | SetTensorElementType | Standalone creation not exposed |
| ➖ | SetDimensions | Standalone creation not exposed |
| ✅ | GetTensorElementType | `TypeInfo.element_type` |
| ✅ | GetDimensionsCount | Used internally by `TypeInfo.shape` |
| ✅ | GetDimensions | `TypeInfo.shape` |
| ✅ | GetSymbolicDimensions | `TypeInfo.dimensions` |
| ✅ | GetTensorShapeElementCount | Used internally |
| ✅ | GetTensorTypeAndShape | Used internally by `Value.GetType()`, `Value.GetShape()` |
| ✅ | GetTypeInfo | Used internally by `Session.get_*_info()` |
| ✅ | GetValueType | `Value.value_type` |
| ✅ | CreateMemoryInfo | `MemoryInfo(name, allocator_type, device_id, mem_type)` |
| ✅ | CreateCpuMemoryInfo | `MemoryInfo()` (default constructor) |
| ✅ | CompareMemoryInfo | `MemoryInfo.__eq__()` |
| ✅ | MemoryInfoGetName | `MemoryInfo.name` |
| ✅ | MemoryInfoGetId | `MemoryInfo.device_id` |
| ✅ | MemoryInfoGetMemType | `MemoryInfo.mem_type` |
| ✅ | MemoryInfoGetType | `MemoryInfo.allocator_type` |
| ➖ | AllocatorAlloc | Allocator management not in scope |
| ➖ | AllocatorFree | Allocator management not in scope |
| ➖ | AllocatorGetInfo | Allocator management not in scope |
| ✅ | GetAllocatorWithDefaultOptions | Used internally (`GetAllocator()`) |
| ✅ | AddFreeDimensionOverride | `SessionOptions.add_free_dimension_override()` |
| ✅ | GetValue | `Value.__getitem__()` (map/sequence element access) |
| ✅ | GetValueCount | `Value.__len__()` |
| ✅ | CreateValue | Used internally for map/sequence construction |
| ➖ | CreateOpaqueValue | Opaque type not exposed |
| ➖ | GetOpaqueValue | Opaque type not exposed |
| ➖ | KernelInfoGetAttribute_float | Custom op / kernel info not in scope |
| ➖ | KernelInfoGetAttribute_int64 | Custom op / kernel info not in scope |
| ➖ | KernelInfoGetAttribute_string | Custom op / kernel info not in scope |
| ➖ | KernelContext_GetInputCount | Custom op / kernel context not in scope |
| ➖ | KernelContext_GetOutputCount | Custom op / kernel context not in scope |
| ➖ | KernelContext_GetInput | Custom op / kernel context not in scope |
| ➖ | KernelContext_GetOutput | Custom op / kernel context not in scope |
| ✅ | ReleaseEnv | `Env` destructor (via RAII) |
| ✅ | ReleaseStatus | `Status` destructor (via RAII) |
| ✅ | ReleaseMemoryInfo | `MemoryInfo` destructor (via RAII) |
| ✅ | ReleaseSession | `Session` destructor (via RAII) |
| ✅ | ReleaseValue | `Value::State` destructor (via RAII) |
| ✅ | ReleaseRunOptions | `RunOptions` destructor (via RAII) |
| ✅ | ReleaseTypeInfo | `TypeInfo` destructor (via RAII) |
| ✅ | ReleaseTensorTypeAndShapeInfo | Used internally (not owned; borrowed pointer) |
| ✅ | ReleaseSessionOptions | `SessionOptions` destructor (via RAII) |
| ➖ | ReleaseCustomOpDomain | Custom op implementation not in scope |
| ✅ | GetDenotationFromTypeInfo | `TypeInfo.denotation` |
| ✅ | CastTypeInfoToMapTypeInfo | Used internally by `TypeInfo.map_key_type` / `map_value_type` |
| ✅ | CastTypeInfoToSequenceTypeInfo | Used internally by `TypeInfo.sequence_element_type` |
| ✅ | GetMapKeyType | `TypeInfo.map_key_type` |
| ✅ | GetMapValueType | `TypeInfo.map_value_type` |
| ✅ | GetSequenceElementType | `TypeInfo.sequence_element_type` |
| ✅ | ReleaseMapTypeInfo | Used internally (borrowed pointer, not released separately) |
| ✅ | ReleaseSequenceTypeInfo | Used internally (borrowed pointer, not released separately) |
| ✅ | SessionEndProfiling | `Session.end_profiling()` |
| ✅ | SessionGetModelMetadata | `Session.get_model_metadata()` |
| ✅ | ModelMetadataGetProducerName | `ModelMetadata.producer_name` |
| ✅ | ModelMetadataGetGraphName | `ModelMetadata.graph_name` |
| ✅ | ModelMetadataGetDomain | `ModelMetadata.domain` |
| ✅ | ModelMetadataGetDescription | `ModelMetadata.description` |
| ✅ | ModelMetadataLookupCustomMetadataMap | `ModelMetadata.lookup_custom_metadata()` |
| ✅ | ModelMetadataGetVersion | `ModelMetadata.version` |
| ✅ | ReleaseModelMetadata | `ModelMetadata` destructor (via RAII) |
| ✅ | CreateEnvWithGlobalThreadPools | `ortpy.create_env(threading_options=...)` |
| ✅ | DisablePerSessionThreads | `SessionOptions.disable_per_session_threads()` |
| ✅ | CreateThreadingOptions | `ThreadingOptions()` |
| ✅ | ReleaseThreadingOptions | `ThreadingOptions` destructor (via RAII) |
| ✅ | ModelMetadataGetCustomMetadataMapKeys | Used internally by `ModelMetadata.custom_metadata_map` |
| ✅ | AddFreeDimensionOverrideByName | `SessionOptions.add_free_dimension_override_by_name()` |
| ✅ | GetAvailableProviders | `ortpy.get_available_providers()` |
| ✅ | ReleaseAvailableProviders | Used internally |
| ✅ | GetStringTensorElementLength | Used internally by `Value.get_strings()` |
| ✅ | GetStringTensorElement | Used internally by `Value.get_strings()` |
| ✅ | FillStringTensorElement | Used internally by `Value.from_strings()` |
| ✅ | AddSessionConfigEntry | `SessionOptions.add_session_config_entry()` |
| ➖ | CreateAllocator | Allocator management not in scope |
| ➖ | ReleaseAllocator | Allocator management not in scope |
| ✅ | RunWithBinding | `Session.run_with_binding()` |
| ✅ | CreateIoBinding | `Session.create_io_binding()` |
| ✅ | ReleaseIoBinding | `IoBinding` destructor (via RAII) |
| ✅ | BindInput | `IoBinding.bind_input()` |
| ✅ | BindOutput | `IoBinding.bind_output()` |
| ✅ | BindOutputToDevice | `IoBinding.bind_output_to_device()` |
| ✅ | GetBoundOutputNames | Used internally by `IoBinding.get_outputs()` |
| ✅ | GetBoundOutputValues | Used internally by `IoBinding.get_outputs()` |
| ✅ | ClearBoundInputs | `IoBinding.clear_inputs()` |
| ✅ | ClearBoundOutputs | `IoBinding.clear_outputs()` |
| ➖ | TensorAt | Low-level typed element access not exposed |
| ➖ | CreateAndRegisterAllocator | Allocator management not in scope |
| ➖ | SetLanguageProjection | Internal ORT mechanism |
| ✅ | SessionGetProfilingStartTimeNs | `Session.get_profiling_start_time_ns()` |
| ✅ | SetGlobalIntraOpNumThreads | `ThreadingOptions.set_intra_op_num_threads()` |
| ✅ | SetGlobalInterOpNumThreads | `ThreadingOptions.set_inter_op_num_threads()` |
| ✅ | SetGlobalSpinControl | `ThreadingOptions.set_spin_control()` |
| ✅ | AddInitializer | `SessionOptions.add_initializer()` |
| ✅ | CreateEnvWithCustomLoggerAndGlobalThreadPools | `ortpy.create_env(logging_function=..., threading_options=...)` |
| ➖ | SessionOptionsAppendExecutionProvider_CUDA | Legacy EP API; use `append_execution_provider()` |
| ➖ | SessionOptionsAppendExecutionProvider_ROCM | Legacy EP API; use `append_execution_provider()` |
| ➖ | SessionOptionsAppendExecutionProvider_OpenVINO | Legacy EP API; use `append_execution_provider()` |
| ✅ | SetGlobalDenormalAsZero | `ThreadingOptions.set_denormal_as_zero()` |
| ➖ | CreateArenaCfg | Arena configuration not in scope |
| ➖ | ReleaseArenaCfg | Arena configuration not in scope |
| ✅ | ModelMetadataGetGraphDescription | `ModelMetadata.graph_description` |
| ➖ | SessionOptionsAppendExecutionProvider_TensorRT | Legacy EP API; use `append_execution_provider()` |
| ❌ | SetCurrentGpuDeviceId | GPU device management not exposed |
| ❌ | GetCurrentGpuDeviceId | GPU device management not exposed |
| ➖ | KernelInfoGetAttributeArray_float | Custom op / kernel info not in scope |
| ➖ | KernelInfoGetAttributeArray_int64 | Custom op / kernel info not in scope |
| ➖ | CreateArenaCfgV2 | Arena configuration not in scope |
| ✅ | AddRunConfigEntry | `RunOptions.add_run_config_entry()` |
| ❌ | CreatePrepackedWeightsContainer | Prepacked weights not exposed |
| ❌ | ReleasePrepackedWeightsContainer | Prepacked weights not exposed |
| ❌ | CreateSessionWithPrepackedWeightsContainer | Prepacked weights not exposed |
| ❌ | CreateSessionFromArrayWithPrepackedWeightsContainer | Prepacked weights not exposed |
| ➖ | SessionOptionsAppendExecutionProvider_TensorRT_V2 | Legacy EP API; use `append_execution_provider()` |
| ➖ | CreateTensorRTProviderOptions | Legacy EP API; use `append_execution_provider()` |
| ➖ | UpdateTensorRTProviderOptions | Legacy EP API; use `append_execution_provider()` |
| ➖ | GetTensorRTProviderOptionsAsString | Legacy EP API; use `append_execution_provider()` |
| ➖ | ReleaseTensorRTProviderOptions | Legacy EP API; use `append_execution_provider()` |
| ✅ | EnableOrtCustomOps | `SessionOptions.enable_ort_custom_ops()` |
| ➖ | RegisterAllocator | Allocator management not in scope |
| ➖ | UnregisterAllocator | Allocator management not in scope |
| ❌ | IsSparseTensor | Sparse tensor not in scope |
| ❌ | CreateSparseTensorAsOrtValue | Sparse tensor not in scope |
| ❌ | FillSparseTensorCoo | Sparse tensor not in scope |
| ❌ | FillSparseTensorCsr | Sparse tensor not in scope |
| ❌ | FillSparseTensorBlockSparse | Sparse tensor not in scope |
| ❌ | CreateSparseTensorWithValuesAsOrtValue | Sparse tensor not in scope |
| ❌ | UseCooIndices | Sparse tensor not in scope |
| ❌ | UseCsrIndices | Sparse tensor not in scope |
| ❌ | UseBlockSparseIndices | Sparse tensor not in scope |
| ❌ | GetSparseTensorFormat | Sparse tensor not in scope |
| ❌ | GetSparseTensorValuesTypeAndShape | Sparse tensor not in scope |
| ❌ | GetSparseTensorValues | Sparse tensor not in scope |
| ❌ | GetSparseTensorIndicesTypeShape | Sparse tensor not in scope |
| ❌ | GetSparseTensorIndices | Sparse tensor not in scope |
| ✅ | HasValue | `Value.has_value` |
| ➖ | KernelContext_GetGPUComputeStream | Custom op / kernel context not in scope |
| ✅ | GetTensorMemoryInfo | `Value.get_tensor_memory_info()` |
| ➖ | GetExecutionProviderApi | EP implementation not in scope |
| ➖ | SessionOptionsSetCustomCreateThreadFn | Custom threading not exposed |
| ➖ | SessionOptionsSetCustomThreadCreationOptions | Custom threading not exposed |
| ➖ | SessionOptionsSetCustomJoinThreadFn | Custom threading not exposed |
| ➖ | SetGlobalCustomCreateThreadFn | Custom threading not exposed |
| ➖ | SetGlobalCustomThreadCreationOptions | Custom threading not exposed |
| ➖ | SetGlobalCustomJoinThreadFn | Custom threading not exposed |
| ✅ | SynchronizeBoundInputs | `IoBinding.synchronize_inputs()` |
| ✅ | SynchronizeBoundOutputs | `IoBinding.synchronize_outputs()` |
| ➖ | KernelInfoGetAttribute_tensor | Custom op / kernel info not in scope |
| ❌ | GetResizedStringTensorElementBuffer | Low-level buffer management not exposed |
| ➖ | CreateAndRegisterAllocatorV2 | Allocator management not in scope |
| ➖ | RunAsync | Async execution not exposed |
| ✅ | ReleaseLoraAdapter | `LoraAdapter` destructor (via RAII) |
| ❌ | ReleaseSyncStream | SyncStream not exposed |

## OrtApi — Version 1.11

| | API | Detail |
|---|---|---|
| ➖ | SessionOptionsAppendExecutionProvider_CUDA_V2 | Legacy EP API; use `append_execution_provider()` |
| ➖ | CreateCUDAProviderOptions | Legacy EP API; use `append_execution_provider()` |
| ➖ | UpdateCUDAProviderOptions | Legacy EP API; use `append_execution_provider()` |
| ➖ | GetCUDAProviderOptionsAsString | Legacy EP API; use `append_execution_provider()` |
| ➖ | ReleaseCUDAProviderOptions | Legacy EP API; use `append_execution_provider()` |
| ➖ | SessionOptionsAppendExecutionProvider_MIGraphX | Legacy EP API; use `append_execution_provider()` |

## OrtApi — Version 1.12

| | API | Detail |
|---|---|---|
| ✅ | AddExternalInitializers | `SessionOptions.add_external_initializers()` |
| ❌ | CreateOpAttr | Op invocation not in scope |
| ❌ | ReleaseOpAttr | Op invocation not in scope |
| ❌ | CreateOp | Op invocation not in scope |
| ❌ | InvokeOp | Op invocation not in scope |
| ❌ | ReleaseOp | Op invocation not in scope |
| ✅ | SessionOptionsAppendExecutionProvider | `SessionOptions.append_execution_provider()` |
| ➖ | CopyKernelInfo | Custom op / kernel info not in scope |
| ➖ | ReleaseKernelInfo | Custom op / kernel info not in scope |

## OrtApi — Version 1.13

| | API | Detail |
|---|---|---|
| ➖ | GetTrainingApi | Training API not in scope |
| ➖ | SessionOptionsAppendExecutionProvider_CANN | Legacy EP API; use `append_execution_provider()` |
| ➖ | CreateCANNProviderOptions | Legacy EP API; use `append_execution_provider()` |
| ➖ | UpdateCANNProviderOptions | Legacy EP API; use `append_execution_provider()` |
| ➖ | GetCANNProviderOptionsAsString | Legacy EP API; use `append_execution_provider()` |
| ➖ | ReleaseCANNProviderOptions | Legacy EP API; use `append_execution_provider()` |

## OrtApi — Version 1.14

| | API | Detail |
|---|---|---|
| ✅ | MemoryInfoGetDeviceType | `MemoryInfo.device_type` |
| ✅ | UpdateEnvWithCustomLogLevel | `ortpy.set_log_level()` |
| ✅ | SetGlobalIntraOpThreadAffinity | `ThreadingOptions.set_intra_op_thread_affinity()` |
| ✅ | RegisterCustomOpsLibrary_V2 | `SessionOptions.register_custom_ops_library_v2()` |
| ✅ | RegisterCustomOpsUsingFunction | `SessionOptions.register_custom_ops_using_function()` |
| ➖ | KernelInfo_GetInputCount | Custom op / kernel info not in scope |
| ➖ | KernelInfo_GetOutputCount | Custom op / kernel info not in scope |
| ➖ | KernelInfo_GetInputName | Custom op / kernel info not in scope |
| ➖ | KernelInfo_GetOutputName | Custom op / kernel info not in scope |
| ➖ | KernelInfo_GetInputTypeInfo | Custom op / kernel info not in scope |
| ➖ | KernelInfo_GetOutputTypeInfo | Custom op / kernel info not in scope |
| ✅ | HasSessionConfigEntry | `SessionOptions.has_session_config_entry()` |
| ✅ | GetSessionConfigEntry | `SessionOptions.get_session_config_entry()` |

## OrtApi — Version 1.15

| | API | Detail |
|---|---|---|
| ➖ | SessionOptionsAppendExecutionProvider_Dnnl | Legacy EP API; use `append_execution_provider()` |
| ➖ | CreateDnnlProviderOptions | Legacy EP API; use `append_execution_provider()` |
| ➖ | UpdateDnnlProviderOptions | Legacy EP API; use `append_execution_provider()` |
| ➖ | GetDnnlProviderOptionsAsString | Legacy EP API; use `append_execution_provider()` |
| ➖ | ReleaseDnnlProviderOptions | Legacy EP API; use `append_execution_provider()` |
| ➖ | KernelInfo_GetNodeName | Custom op / kernel info not in scope |
| ➖ | KernelInfo_GetLogger | Custom op / kernel info not in scope |
| ➖ | KernelContext_GetLogger | Custom op / kernel context not in scope |
| ❌ | Logger_LogMessage | Logger API not exposed |
| ❌ | Logger_GetLoggingSeverityLevel | Logger API not exposed |
| ➖ | KernelInfoGetConstantInput_tensor | Custom op / kernel info not in scope |
| ✅ | CastTypeInfoToOptionalTypeInfo | Used internally by `TypeInfo.optional_contained_type` |
| ✅ | GetOptionalContainedTypeInfo | `TypeInfo.optional_contained_type` |
| ➖ | KernelContext_GetAllocator | Custom op / kernel context not in scope |
| ➖ | GetBuildInfoString | Not exposed |

## OrtApi — Version 1.16

| | API | Detail |
|---|---|---|
| ➖ | CreateROCMProviderOptions | Legacy EP API; use `append_execution_provider()` |
| ➖ | UpdateROCMProviderOptions | Legacy EP API; use `append_execution_provider()` |
| ➖ | GetROCMProviderOptionsAsString | Legacy EP API; use `append_execution_provider()` |
| ➖ | ReleaseROCMProviderOptions | Legacy EP API; use `append_execution_provider()` |
| ➖ | UpdateTensorRTProviderOptionsWithValue | Legacy EP API; use `append_execution_provider()` |
| ➖ | GetTensorRTProviderOptionsByName | Legacy EP API; use `append_execution_provider()` |
| ➖ | UpdateCUDAProviderOptionsWithValue | Legacy EP API; use `append_execution_provider()` |
| ➖ | GetCUDAProviderOptionsByName | Legacy EP API; use `append_execution_provider()` |
| ➖ | KernelContext_GetResource | Custom op / kernel context not in scope |

## OrtApi — Version 1.17

| | API | Detail |
|---|---|---|
| ✅ | SetUserLoggingFunction | `SessionOptions.set_user_logging_function()` |
| ❌ | ShapeInferContext_GetInputCount | Shape inference not in scope |
| ❌ | ShapeInferContext_GetInputTypeShape | Shape inference not in scope |
| ❌ | ShapeInferContext_GetAttribute | Shape inference not in scope |
| ❌ | ShapeInferContext_SetOutputTypeShape | Shape inference not in scope |
| ❌ | SetSymbolicDimensions | Standalone creation not exposed |
| ❌ | ReadOpAttr | Op invocation not in scope |
| ✅ | SetDeterministicCompute | `SessionOptions.set_deterministic_compute()` |
| ➖ | KernelContext_ParallelFor | Custom op / kernel context not in scope |
| ➖ | SessionOptionsAppendExecutionProvider_OpenVINO_V2 | Legacy EP API; use `append_execution_provider()` |

## OrtApi — Version 1.18

| | API | Detail |
|---|---|---|
| ➖ | SessionOptionsAppendExecutionProvider_VitisAI | Legacy EP API; use `append_execution_provider()` |
| ➖ | KernelContext_GetScratchBuffer | Custom op / kernel context not in scope |
| ➖ | KernelInfoGetAllocator | Custom op / kernel info not in scope |
| ✅ | AddExternalInitializersFromFilesInMemory | `SessionOptions.add_external_initializers_from_files_in_memory()` |

## OrtApi — Version 1.20

| | API | Detail |
|---|---|---|
| ✅ | CreateLoraAdapter | `LoraAdapter(adapter_file_path)` |
| ✅ | CreateLoraAdapterFromArray | `LoraAdapter(adapter_bytes)` |
| ✅ | RunOptionsAddActiveLoraAdapter | `RunOptions.add_active_lora_adapter()` |
| ❌ | SetEpDynamicOptions | EP dynamic options not exposed |

## OrtApi — Version 1.22

| | API | Detail |
|---|---|---|
| ❌ | ReleaseValueInfo | Graph inspection not in scope |
| ❌ | ReleaseNode | Graph inspection not in scope |
| ❌ | ReleaseGraph | Graph inspection not in scope |
| ❌ | ReleaseModel | Graph inspection not in scope |
| ❌ | GetValueInfoName | Graph inspection not in scope |
| ❌ | GetValueInfoTypeInfo | Graph inspection not in scope |
| ❌ | GetModelEditorApi | Model editor not in scope |
| ❌ | CreateTensorWithDataAndDeleterAsOrtValue | Custom deleter tensors not exposed |
| ✅ | SessionOptionsSetLoadCancellationFlag | `SessionOptions.set_load_cancellation_flag()` |
| ✅ | GetCompileApi | Used internally by `ModelCompilationOptions` |
| ✅ | CreateKeyValuePairs | Used internally by EP registration helpers |
| ✅ | AddKeyValuePair | Used internally by EP registration helpers |
| ➖ | GetKeyValue | Not directly exposed |
| ➖ | GetKeyValuePairs | Not directly exposed |
| ➖ | RemoveKeyValuePair | Not directly exposed |
| ✅ | ReleaseKeyValuePairs | Used internally (via RAII) |
| ✅ | RegisterExecutionProviderLibrary | `ortpy.register_execution_provider_library()` |
| ✅ | UnregisterExecutionProviderLibrary | `ortpy.unregister_execution_provider_library()` |
| ✅ | GetEpDevices | `ortpy.get_ep_devices()` |
| ✅ | SessionOptionsAppendExecutionProvider_V2 | `SessionOptions.append_execution_provider_v2()` |
| ✅ | SessionOptionsSetEpSelectionPolicy | `SessionOptions.set_ep_selection_policy()` |
| ✅ | SessionOptionsSetEpSelectionPolicyDelegate | `SessionOptions.set_ep_selection_policy_delegate()` |
| ✅ | HardwareDevice_Type | `HardwareDevice.type` |
| ✅ | HardwareDevice_VendorId | `HardwareDevice.vendor_id` |
| ✅ | HardwareDevice_Vendor | `HardwareDevice.vendor` |
| ✅ | HardwareDevice_DeviceId | `HardwareDevice.device_id` |
| ✅ | HardwareDevice_Metadata | `HardwareDevice.metadata` |
| ✅ | EpDevice_EpName | `EpDevice.ep_name` |
| ✅ | EpDevice_EpVendor | `EpDevice.ep_vendor` |
| ✅ | EpDevice_EpMetadata | `EpDevice.ep_metadata` |
| ✅ | EpDevice_EpOptions | `EpDevice.ep_options` |
| ✅ | EpDevice_Device | `EpDevice.device` |
| ➖ | GetEpApi | EP implementation not in scope |

## OrtApi — Version 1.23

| | API | Detail |
|---|---|---|
| ✅ | GetTensorSizeInBytes | `Value.get_tensor_size_in_bytes()` |
| ➖ | AllocatorGetStats | Allocator management not in scope |
| ❌ | CreateMemoryInfo_V2 | V2 memory info creation not exposed |
| ❌ | MemoryInfoGetDeviceMemType | Not exposed |
| ❌ | MemoryInfoGetVendorId | Not exposed |
| ❌ | ValueInfo_GetValueProducer | Graph inspection not in scope |
| ❌ | ValueInfo_GetValueNumConsumers | Graph inspection not in scope |
| ❌ | ValueInfo_GetValueConsumers | Graph inspection not in scope |
| ❌ | ValueInfo_GetInitializerValue | Graph inspection not in scope |
| ❌ | ValueInfo_GetExternalInitializerInfo | Graph inspection not in scope |
| ❌ | ValueInfo_IsRequiredGraphInput | Graph inspection not in scope |
| ❌ | ValueInfo_IsOptionalGraphInput | Graph inspection not in scope |
| ❌ | ValueInfo_IsGraphOutput | Graph inspection not in scope |
| ❌ | ValueInfo_IsConstantInitializer | Graph inspection not in scope |
| ❌ | ValueInfo_IsFromOuterScope | Graph inspection not in scope |
| ❌ | Graph_GetName | Graph inspection not in scope |
| ❌ | Graph_GetModelPath | Graph inspection not in scope |
| ❌ | Graph_GetOnnxIRVersion | Graph inspection not in scope |
| ❌ | Graph_GetNumOperatorSets | Graph inspection not in scope |
| ❌ | Graph_GetOperatorSets | Graph inspection not in scope |
| ❌ | Graph_GetNumInputs | Graph inspection not in scope |
| ❌ | Graph_GetInputs | Graph inspection not in scope |
| ❌ | Graph_GetNumOutputs | Graph inspection not in scope |
| ❌ | Graph_GetOutputs | Graph inspection not in scope |
| ❌ | Graph_GetNumInitializers | Graph inspection not in scope |
| ❌ | Graph_GetInitializers | Graph inspection not in scope |
| ❌ | Graph_GetNumNodes | Graph inspection not in scope |
| ❌ | Graph_GetNodes | Graph inspection not in scope |
| ❌ | Graph_GetParentNode | Graph inspection not in scope |
| ❌ | Graph_GetGraphView | Graph inspection not in scope |
| ❌ | Node_GetId | Graph inspection not in scope |
| ❌ | Node_GetName | Graph inspection not in scope |
| ❌ | Node_GetOperatorType | Graph inspection not in scope |
| ❌ | Node_GetDomain | Graph inspection not in scope |
| ❌ | Node_GetSinceVersion | Graph inspection not in scope |
| ❌ | Node_GetNumInputs | Graph inspection not in scope |
| ❌ | Node_GetInputs | Graph inspection not in scope |
| ❌ | Node_GetNumOutputs | Graph inspection not in scope |
| ❌ | Node_GetOutputs | Graph inspection not in scope |
| ❌ | Node_GetNumImplicitInputs | Graph inspection not in scope |
| ❌ | Node_GetImplicitInputs | Graph inspection not in scope |
| ❌ | Node_GetNumAttributes | Graph inspection not in scope |
| ❌ | Node_GetAttributes | Graph inspection not in scope |
| ❌ | Node_GetAttributeByName | Graph inspection not in scope |
| ❌ | OpAttr_GetTensorAttributeAsOrtValue | Graph inspection not in scope |
| ❌ | OpAttr_GetType | Graph inspection not in scope |
| ❌ | OpAttr_GetName | Graph inspection not in scope |
| ❌ | Node_GetNumSubgraphs | Graph inspection not in scope |
| ❌ | Node_GetSubgraphs | Graph inspection not in scope |
| ❌ | Node_GetGraph | Graph inspection not in scope |
| ❌ | Node_GetEpName | Graph inspection not in scope |
| ❌ | ReleaseExternalInitializerInfo | Graph inspection not in scope |
| ❌ | ExternalInitializerInfo_GetFilePath | Graph inspection not in scope |
| ❌ | ExternalInitializerInfo_GetFileOffset | Graph inspection not in scope |
| ❌ | ExternalInitializerInfo_GetByteSize | Graph inspection not in scope |
| ✅ | GetRunConfigEntry | `RunOptions.get_run_config_entry()` |
| ✅ | EpDevice_MemoryInfo | `EpDevice.get_memory_info()` |
| ➖ | CreateSharedAllocator | Allocator management not in scope |
| ➖ | GetSharedAllocator | Allocator management not in scope |
| ➖ | ReleaseSharedAllocator | Allocator management not in scope |
| ❌ | GetTensorData | Low-level pointer access; use `Value.numpy()` instead |
| ✅ | GetSessionOptionsConfigEntries | `SessionOptions.get_session_config_entries()` |
| ✅ | SessionGetMemoryInfoForInputs | `Session.get_memory_info_for_inputs()` |
| ✅ | SessionGetMemoryInfoForOutputs | `Session.get_memory_info_for_outputs()` |
| ✅ | SessionGetEpDeviceForInputs | `Session.get_ep_device_for_inputs()` |
| ❌ | CreateSyncStreamForEpDevice | SyncStream not exposed |
| ❌ | SyncStream_GetHandle | SyncStream not exposed |
| ❌ | CopyTensors | Tensor copy not exposed |
| ❌ | Graph_GetModelMetadata | Graph inspection not in scope |
| ✅ | GetModelCompatibilityForEpDevices | `ortpy.get_model_compatibility_for_ep_devices()` |
| ❌ | CreateExternalInitializerInfo | Graph inspection not in scope |

## OrtApi — Version 1.24

| | API | Detail |
|---|---|---|
| ❌ | TensorTypeAndShape_HasShape | Not exposed |
| ➖ | KernelInfo_GetConfigEntries | Custom op / kernel info not in scope |
| ➖ | KernelInfo_GetOperatorDomain | Custom op / kernel info not in scope |
| ➖ | KernelInfo_GetOperatorType | Custom op / kernel info not in scope |
| ➖ | KernelInfo_GetOperatorSinceVersion | Custom op / kernel info not in scope |
| ❌ | GetInteropApi | Interop API not in scope |
| ✅ | SessionGetEpDeviceForOutputs | `Session.get_ep_device_for_outputs()` (guarded `ORT_API_VERSION >= 24`) |
| ✅ | GetNumHardwareDevices | Used internally by `ortpy.get_hardware_devices()` (guarded) |
| ✅ | GetHardwareDevices | `ortpy.get_hardware_devices()` (guarded `ORT_API_VERSION >= 24`) |
| ✅ | GetHardwareDeviceEpIncompatibilityDetails | `ortpy.get_hardware_device_ep_incompatibility_details()` (guarded) |
| ✅ | DeviceEpIncompatibilityDetails_GetReasonsBitmask | `DeviceEpIncompatibilityInfo.reasons_bitmask` (guarded) |
| ✅ | DeviceEpIncompatibilityDetails_GetNotes | `DeviceEpIncompatibilityInfo.notes` (guarded) |
| ✅ | DeviceEpIncompatibilityDetails_GetErrorCode | `DeviceEpIncompatibilityInfo.error_code` (guarded) |
| ✅ | ReleaseDeviceEpIncompatibilityDetails | Used internally (guarded) |
| ✅ | GetCompatibilityInfoFromModel | `ortpy.get_compatibility_info_from_model()` (guarded) |
| ✅ | GetCompatibilityInfoFromModelBytes | `ortpy.get_compatibility_info_from_model_bytes()` (guarded) |
| ✅ | CreateEnvWithOptions | `ortpy.create_env(...)` (guarded `ORT_API_VERSION >= 24`) |
| ✅ | Session_GetEpGraphAssignmentInfo | `Session.get_ep_graph_assignment_info()` (guarded `ORT_API_VERSION >= 24`) |
| ✅ | EpAssignedSubgraph_GetEpName | `EpAssignedSubgraph.ep_name` (guarded) |
| ✅ | EpAssignedSubgraph_GetNodes | `EpAssignedSubgraph.nodes` (guarded) |
| ✅ | EpAssignedNode_GetName | `EpAssignedNode.name` (guarded) |
| ✅ | EpAssignedNode_GetDomain | `EpAssignedNode.domain` (guarded) |
| ✅ | EpAssignedNode_GetOperatorType | `EpAssignedNode.operator_type` (guarded) |
| ❌ | RunOptionsSetSyncStream | SyncStream not exposed |
| ❌ | GetTensorElementTypeAndShapeDataReference | Low-level reference not exposed |

---

## OrtCompileApi (Version 1.22+)

| | API | Detail |
|---|---|---|
| ✅ | ReleaseModelCompilationOptions | `ModelCompilationOptions` destructor (via RAII) |
| ✅ | CreateModelCompilationOptionsFromSessionOptions | `SessionOptions.create_model_compilation_options()` |
| ✅ | ModelCompilationOptions_SetInputModelPath | `ModelCompilationOptions.set_input_model_path()` |
| ✅ | ModelCompilationOptions_SetInputModelFromBuffer | `ModelCompilationOptions.set_input_model_from_buffer()` |
| ✅ | ModelCompilationOptions_SetOutputModelPath | Used internally by `compile_model_to_file()` |
| ✅ | ModelCompilationOptions_SetOutputModelExternalInitializersFile | `ModelCompilationOptions.set_output_model_external_initializers_file()` |
| ✅ | ModelCompilationOptions_SetOutputModelBuffer | Used internally by `compile_model_to_buffer()` |
| ✅ | ModelCompilationOptions_SetEpContextEmbedMode | `ModelCompilationOptions.set_ep_context_embed_mode()` |
| ✅ | CompileModel | Used internally by `compile_model_to_file()` / `compile_model_to_buffer()` |
| ✅ | ModelCompilationOptions_SetFlags | `ModelCompilationOptions.set_flags()` |
| ✅ | ModelCompilationOptions_SetEpContextBinaryInformation | `ModelCompilationOptions.set_ep_context_binary_information()` |
| ✅ | ModelCompilationOptions_SetGraphOptimizationLevel | `ModelCompilationOptions.set_graph_optimization_level()` |
| ✅ | ModelCompilationOptions_SetOutputModelWriteFunc | `ModelCompilationOptions.set_output_model_write_func()` |
| ❌ | ModelCompilationOptions_SetOutputModelGetInitializerLocationFunc | Custom initializer location callback not exposed |

---

## OrtInteropApi (Version 1.24)

| | API | Detail |
|---|---|---|
| ❌ | CreateExternalResourceImporterForDevice | GPU interop not in scope |
| ❌ | ReleaseExternalResourceImporter | GPU interop not in scope |
| ❌ | CanImportMemory | GPU interop not in scope |
| ❌ | ImportMemory | GPU interop not in scope |
| ❌ | ReleaseExternalMemoryHandle | GPU interop not in scope |
| ❌ | CreateTensorFromMemory | GPU interop not in scope |
| ❌ | CanImportSemaphore | GPU interop not in scope |
| ❌ | ImportSemaphore | GPU interop not in scope |
| ❌ | ReleaseExternalSemaphoreHandle | GPU interop not in scope |
| ❌ | WaitSemaphore | GPU interop not in scope |
| ❌ | SignalSemaphore | GPU interop not in scope |

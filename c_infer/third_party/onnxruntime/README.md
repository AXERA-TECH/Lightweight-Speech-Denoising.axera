# ONNX Runtime placeholder

The x86 C inference target expects ONNX Runtime to be placed here:

```text
c_infer/third_party/onnxruntime/
+-- include/
|   `-- onnxruntime_c_api.h
`-- lib/
    `-- libonnxruntime.so
```

The full ONNX Runtime package is not committed because it is large. Download or unpack the release artifact here before building x86 C inference.

For local development, this directory can be populated from `onnxruntime-linux-x64-1.20.1` or a compatible ONNX Runtime Linux x64 release.

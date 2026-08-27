# infe.rs Kotlin Multiplatform module

## Modules

```toml
[versions]
infers = "..."

[libraries]
infers-core = { module = "dev.stashy.infers:infers", version.ref = "infers" }
infers-portable = { module = "dev.stashy.infers:infers-portable", version.ref = "infers" }
infers-xnnpack = { module = "dev.stashy.infers:infers-xnnpack", version.ref = "infers" }
infers-vulkan = { module = "dev.stashy.infers:infers-vulkan", version.ref = "infers" }
```

## Usage

```kotlin
val cameraSource = TODO()
val outputFlow = MutableStateFlow()
val backend = Backend() // initializes backend, currently only executorch 
val model = backend.loadModel(Path("model.pte"), XnnpackConfig()) // loads model into executorch to be run via the CPU
val processor = CpuImageProcessor() // image processing utils for the CPU

cameraSource.forEach { frame ->
    inferenceScope { // memory cleanup scope - processing functions are only provided within this scope
        val input = processor.process(frame, options)
        val outputs = session.run(input)
        outputFlow.emit(outputs.first().readFloats())
    }
}

// everything must be closed for proper cleanup
processor.close()
model.close()
backend.close()
```

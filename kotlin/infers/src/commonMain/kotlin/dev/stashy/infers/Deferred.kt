package dev.stashy.infers

/** Unmaterialized image placement on device [D]. */
public interface Deferred<D : Device> : AutoCloseable

/** Unmaterialized preprocess/tensor work on device [D]. */
public interface Pending<D : Device> : AutoCloseable {
    public val shape: TensorShape
    public val dtype: DataType
}

import pyarrow
import numpy as np
import cupy as cp
import cudf
import rmm

import time

rmm.reinitialize(
    pool_allocator=True,
    managed_memory=True,
)

print("GPU:",
      cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8'))

data = pyarrow.Table.from_arrays(
    [np.random.randint(1, 50, 250000000, dtype=np.int32),
     np.random.randint(1, 10, 250000000, dtype=np.int32)],
    ["key", "val"]
    )

start = time.perf_counter()

start_init = time.perf_counter()
data = cudf.DataFrame.from_arrow(data)
end_init = time.perf_counter()

start_aggregate = time.perf_counter()
data = data.groupby("key").agg({"val": "sum"})
end_aggregate = time.perf_counter()

start_fin = time.perf_counter()
data = data.to_arrow(preserve_index=False)
end_fin = time.perf_counter()

end = time.perf_counter()

print("Initial transfer:", end_init-start_init)
print("Aggregation time GPU:", end_aggregate-start_aggregate)
print("Final transfer:", end_fin-start_fin)
print("Time:", end-start)
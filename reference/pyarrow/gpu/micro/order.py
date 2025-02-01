import numpy as np
import cupy as cp
import cudf
import pyarrow
import rmm

import time

rmm.reinitialize(
    pool_allocator=True,
    managed_memory=True,
)

print("GPU:",
      cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8'))

data = pyarrow.Table.from_arrays(
    [np.random.randint(1, 0xffffffff, 500000000, dtype=np.uint32)],
    ["key"]
    )

start = time.perf_counter()

start_init = time.perf_counter()
data = cudf.DataFrame.from_arrow(data)
end_init = time.perf_counter()

start_order = time.perf_counter()
data = data.sort_values("key")
end_order = time.perf_counter()

start_fin = time.perf_counter()
data = data.to_arrow(preserve_index=False)
end_fin = time.perf_counter()

end = time.perf_counter()

print("Initial transfer:", end_init-start_init)
print("Ordering time GPU:", end_order-start_order)
print("Final transfer:", end_fin-start_fin)
print("Time:", end-start)
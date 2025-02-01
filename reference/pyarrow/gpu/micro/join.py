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

inner = pyarrow.Table.from_arrays(
    [np.arange(1, 256000000, dtype=np.uint32)],
    ["key"]
    )

outer = pyarrow.Table.from_arrays(
    [np.random.randint(1, 256000000, 256000000, dtype=np.uint32)],
    ["key"]
    )

start = time.perf_counter()

start_init = time.perf_counter()
inner = cudf.DataFrame.from_arrow(inner)
outer = cudf.DataFrame.from_arrow(outer)
end_init = time.perf_counter()

start_join = time.perf_counter()
joined = inner.merge(outer)
end_join = time.perf_counter()

start_fin = time.perf_counter()
joined = joined.to_arrow(preserve_index=False)
end_fin = time.perf_counter()

end = time.perf_counter()

print("Initial transfer:", end_init-start_init)
print("Join time GPU:", end_join-start_join)
print("Final transfer:", end_fin-start_fin)
print("Time:", end-start)
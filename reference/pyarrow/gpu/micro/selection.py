import numpy as np
import cupy as cp
import cudf
import pyarrow
import rmm

import time

# rmm.reinitialize(
#     pool_allocator=True,
#     managed_memory=True,
# )

print("GPU:",
      cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8'))

data = pyarrow.Table.from_arrays(
    [np.random.randint(1, 50, 500000000, dtype=np.int32)],
    ["key"]
    )

start = time.perf_counter()

start_init = time.perf_counter()
data = cudf.DataFrame.from_arrow(data)
end_init = time.perf_counter()

start_select = time.perf_counter()
data = data.query("key >= 10 and key <= 20")
end_select = time.perf_counter()

start_fin = time.perf_counter()
data = data.to_arrow(preserve_index=False)
end_fin = time.perf_counter()

end = time.perf_counter()

print("Initial transfer:", end_init-start_init)
print("Selection time GPU:", end_select-start_select)
print("Final transfer:", end_fin-start_fin)
print("Time:", end-start)
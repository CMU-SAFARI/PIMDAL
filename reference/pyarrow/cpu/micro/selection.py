import pyarrow
import pyarrow.compute as pc
import numpy as np
import time

repetitions = 10

def query(data):
    expr = (pc.field("key") >= 10) & (pc.field("key") <= 20)
    res = data.filter(expr)

if __name__ == "__main__":
    data = pyarrow.Table.from_arrays(
        [np.random.randint(1, 50, 2000000000, dtype=np.int32)],
        ["key"]
    )

    start = time.perf_counter()

    for i in range(repetitions):
        query(data)

    end = time.perf_counter()

    print("Selection time CPU:", (end-start) / repetitions)
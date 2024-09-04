#an implementation of the map function kernel in triton for practice
import triton
import triton.language as tl
import torch
import time

if not torch.cuda.is_available():
    raise Exception("CUDA is not available. Exiting.")

@triton.jit
def triton_map(x_ptr: tl.tensor, y_ptr: tl.tensor, B0: tl.constexpr, n: tl.int32):
    # Compute the program ID
    pid = tl.program_id(axis=0)
    # Compute the offset for this thread
    offsets = pid * B0 + tl.arange(0, B0)
    # Create a mask for valid elements
    mask = offsets < n
    # Load x values
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute y values
    y = x / (x - 2.3)
    y = y * y * y
    # Store y values
    tl.store(y_ptr + offsets, y, mask=mask)

GPU_RUNS = 100
N = 75341

device = torch.device("cuda", 0)

X = torch.arange(0, N).to(device)
Y = torch.zeros(N, dtype=torch.float32).to(device)

torch.cuda.synchronize()
start_time = time.perf_counter()
for i in range(GPU_RUNS):
    grid = (triton.cdiv(N, 32),) 
    triton_map[grid](X, Y, B0=32, n=N)
torch.cuda.synchronize()
end_time = time.perf_counter()

total_time = end_time - start_time
mean_time = total_time / GPU_RUNS

print(f"Total time: {total_time:.6f} seconds")
print(f"Mean time per run: {mean_time:.6f} seconds")
gigabytes_per_sec = (2.0 * N * 4.0) / (mean_time * 1000.0)
print(f"Gigabytes per second: {gigabytes_per_sec:.6f}")
print(Y)
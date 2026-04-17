import os
import numpy as np
import torch
import argparse
import minuet
import minuet.nn.functional as MF
from minuet.utils.helpers import generate_kernel_offsets
import torchsparse
import spira.nn.functional as SF

def benchmark_torchsparse(coords, kernel_size, stride):
    # Preallocate hashmap
    kmap = {
        "hashmap_keys": torch.zeros(2 * coords.shape[0], dtype=torch.int64, device='cuda'),
        "hashmap_vals": torch.zeros(2 * coords.shape[0], dtype=torch.int32, device='cuda')
    }
    hashmap = torchsparse.backend.GPUHashTable(kmap["hashmap_keys"], kmap["hashmap_vals"])

    hashmap.insert_coords(coords[:, [1, 2, 3, 0]])
    kernel_volume = torch.prod(kernel_size)
    _ = hashmap.lookup_coords(coords[:, [1, 2, 3, 0]], kernel_size, stride, kernel_volume) - 1


def benchmark_spira(packed_coords, kernel_size):

    _ = SF.build_kmap(packed_coords, packed_coords, kernel_size, 1)


def benchmark_simple_bs(packed_coords, kernel_size):
    
    _ = SF.build_kmap_no_small_z(packed_coords, packed_coords, kernel_size, 1)


def benchmark_minuet(sources, kernel_size):
    offsets = generate_kernel_offsets(
        ndim=3,
        kernel_size=kernel_size,
        source_stride=1,
        dilation=1
    )
    offsets = torch.tensor(offsets, dtype=sources.dtype, device=sources.device)

    _ = MF.query_sorted_index_with_offsets(
        sources=sources,
        targets=sources,
        offsets=offsets,
    )

def main(args):
    # Load voxel coordinates
    tensor_data_np = np.load(args.file)
    print(f"Loaded {tensor_data_np.shape[0]} voxels")
    tensor_data = torch.tensor(tensor_data_np, dtype=torch.int32, device='cuda')

    # Shift coordinates to start from 0
    min_coords = torch.min(tensor_data, dim=0).values
    tensor_data = tensor_data - min_coords

    # Handle kernel size input
    kernel_size = args.kernel_size
    kernel_size = [kernel_size] * 3
    kernel_size_tensor = torch.tensor(kernel_size, dtype=torch.int32, device='cuda')
    stride_tensor = torch.tensor([1, 1, 1], dtype=torch.int32, device='cuda')

    # Prepare library-specific data
    if args.library == "minuet":
        index = MF.build_sorted_index(tensor_data)
        sources = tensor_data[index]
        benchmark_fn = lambda: benchmark_minuet(sources, kernel_size)

    elif args.library == "torchsparse":
        batch = torch.zeros((tensor_data.shape[0], 1), dtype=torch.int32, device="cuda")
        coords = torch.cat([batch, tensor_data], dim=1)
        benchmark_fn = lambda: benchmark_torchsparse(coords, kernel_size_tensor, stride_tensor)

    elif args.library == "simple_bs":
        packed_coords, _ = SF.flat_sort(tensor_data)
        benchmark_fn = lambda: benchmark_simple_bs(packed_coords, kernel_size_tensor)

    elif args.library == "spira":
        packed_coords, _ = SF.flat_sort(tensor_data)
        benchmark_fn = lambda: benchmark_spira(packed_coords, kernel_size)

    # Warm-up
    for _ in range(3):
        benchmark_fn()
    torch.cuda.synchronize()

    # Benchmark
    timings = []
    for _ in range(4):
        torch.cuda.synchronize()
        event1 = torch.cuda.Event(enable_timing=True)
        event2 = torch.cuda.Event(enable_timing=True)
        event1.record()

        benchmark_fn()

        event2.record()
        event2.synchronize()
        timings.append(event1.elapsed_time(event2))

    print(f"{args.library} Benchmark")
    print(f"kernel size: {kernel_size}")
    print(f"Average: {sum(timings)/len(timings):.4f} ms")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Minuet mapping kernel")
    parser.add_argument("--file", type=str, required=True,
                        help="Path to .npy file containing voxel coordinates (N x 3)")
    parser.add_argument("--kernel_size", type=int, help="kernel size", required=True)
    parser.add_argument("--library", type=str, choices=["minuet", "torchsparse", "simple_bs", "spira"],
                        required=True, help="Which library to benchmark")
    main(parser.parse_args())


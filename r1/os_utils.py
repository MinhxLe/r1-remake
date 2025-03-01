import os
import torch


def n_cores() -> int:
    return os.cpu_count()


def print_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = (
                torch.cuda.get_device_properties(i).total_memory / 1024**3
            )  # GB
            reserved_memory = torch.cuda.memory_reserved(i) / 1024**3  # GB
            allocated_memory = torch.cuda.memory_allocated(i) / 1024**3  # GB
            free_memory = total_memory - reserved_memory

            print(f"GPU {i}:")
            print(f"  Total memory: {total_memory:.2f} GB")
            print(f"  Reserved memory: {reserved_memory:.2f} GB")
            print(f"  Allocated memory: {allocated_memory:.2f} GB")
            print(f"  Free memory: {free_memory:.2f} GB")
    else:
        print("CUDA is not available")

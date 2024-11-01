import torch
import time

# Check if CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please run this script on a GPU-enabled machine.")

# Initialize CUDA
device = torch.device('cuda')

# List to hold tensors and prevent garbage collection
leaky_list = []

def create_memory_leak():
    """Simulate a memory leak by creating tensors and holding onto them."""
    print(f"Running on device: {torch.cuda.get_device_name(device)}")

    for i in range(100):
        # Allocate a tensor on the GPU with minimal computation
        tensor = torch.zeros((512, 512), device=device)

        # Simulate light GPU usage with a simple operation
        tensor += 1

        # Retain the tensor reference to simulate a memory leak
        leaky_list.append(tensor)

        # Display current memory usage on the GPU
        print(f"Iteration {i + 1}: Allocated Memory: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")

        # Add delay to make the memory consumption gradual
        time.sleep(1)

if __name__ == "__main__":
    create_memory_leak()

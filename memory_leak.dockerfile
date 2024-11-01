# Use NVIDIA's CUDA base image with PyTorch pre-installed
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Install Python and required packages
RUN apt-get update && apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip3 install torch==2.4.1

# Set the working directory inside the container
WORKDIR /app

# Copy the memory leak script into the container
COPY memory_leak.py .

# Command to run the script
CMD ["python3", "memory_leak.py"]

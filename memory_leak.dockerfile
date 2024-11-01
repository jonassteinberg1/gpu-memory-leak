# Use NVIDIA's CUDA base image with Python 3.9
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Install Python 3.9 and dependencies
RUN apt-get update && \
    apt-get install -y python3.9 python3.9-distutils curl && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as the default Python
RUN ln -s /usr/bin/python3.9 /usr/local/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

# Install PyTorch 2.0 with CUDA 11.8
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118

# Set the working directory inside the container
WORKDIR /app

# Copy the memory leak script into the container
COPY memory_leak.py .

# Command to run the script
CMD ["python3", "memory_leak.py"]

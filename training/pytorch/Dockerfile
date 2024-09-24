# Use the NVIDIA PyTorch image as the base
FROM nvcr.io/nvidia/pytorch:24.08-py3

# Set the working directory in the container
WORKDIR /workspace

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir --timeout=1 -r requirements.txt

RUN apt-get update \
    && apt-get install -y
   
# Set the default command to bash
CMD ["/bin/bash"]

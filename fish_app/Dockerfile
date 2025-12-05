FROM nvcr.io/nvidia/deepstream:7.0-triton-multiarch

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgirepository1.0-dev \
    libcairo2-dev \
    gir1.2-aravis-0.8 \
    libaravis-dev \
    aravis-tools \
    python3-gi \
    python3-dev \
    python3-pip \
    python3-gst-1.0 \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip3 install --upgrade pip
RUN pip3 install opencv-python-headless pyyaml numpy

# Install pyds (DeepStream Python Bindings)
# Note: This usually requires building from source or installing a wheel compatible with the DS version.
# For DS 7.0, we can try to install from the provided wheels in the image if available, or clone deepstream_python_apps
RUN cd /opt/nvidia/deepstream/deepstream/sources/ && \
    if [ -d "deepstream_python_apps" ]; then \
        echo "deepstream_python_apps exists"; \
    else \
        git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git; \
    fi

# Assuming pyds is available or installed. If not, user needs to build it.
# For simplicity, we assume the base image or user handles pyds.

WORKDIR /app
COPY . /app

CMD ["python3", "deepstream_app.py"]

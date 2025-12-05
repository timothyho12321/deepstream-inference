# Use the multi-arch image which supports Jetson (L4T) for DeepStream 7.0
# Use the multi-arch image which supports Jetson (L4T) for DeepStream 7.0
FROM nvcr.io/nvidia/deepstream:7.0-triton-multiarch-multiarch

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    g++ \
    build-essential \
    libglib2.0-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgirepository1.0-dev \
    libcairo2-dev \
    gir1.2-aravis-0.8 \
    libaravis-dev \
    aravis-tools \
    python3-gi \
    python3-gi-cairo \
    python3-dev \
    python3-pip \
    python3-gst-1.0 \
    python3-numpy \
    meson \
    ninja-build \
    gobject-introspection \
    && rm -rf /var/lib/apt/lists/*

# Install python packages (pin numpy to 1.26.x for compatibility)
RUN pip3 install --upgrade pip && \
    pip3 uninstall -y numpy || true && \
    pip3 install --no-cache-dir "numpy==1.26.4" && \
    pip3 install --no-cache-dir opencv-python-headless pyyaml

# Fix missing symlinks for DeepStream libraries (known issue in DS 7.0 container)
RUN cd /opt/nvidia/deepstream/deepstream/lib/ && \
    ln -sf libnvbufsurface.so.1.0.0 libnvbufsurface.so && \
    ln -sf libnvbufsurftransform.so.1.0.0 libnvbufsurftransform.so && \
    ldconfig

# Install pyds (DeepStream Python Bindings) for DS 7.0
# We skip building gst-python because we installed python3-gst-1.0 via apt
RUN cd /opt/nvidia/deepstream/deepstream/sources/ && \
    git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git && \
    cd deepstream_python_apps && \
    git checkout v1.1.11 && \
    git submodule update --init && \
    cd bindings && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    pip3 install ./pyds-*.whl

WORKDIR /app
COPY . /app

CMD ["python3", "deepstream_app.py"]

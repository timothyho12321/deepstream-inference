# Fish Monitoring System (Jetson Orin / Linux)

This application is designed to run on **NVIDIA Jetson Orin** (or x86 Linux with NVIDIA GPU) using DeepStream 7.0.

## Prerequisites (Jetson / Ubuntu)

1.  **NVIDIA JetPack 6.0+** (DeepStream 7.0 requires JetPack 6 on Orin).
2.  **Docker** and **NVIDIA Container Runtime** (usually pre-installed on JetPack).
3.  **Aravis-compatible GigE Cameras** connected and configured with IPs.

## Setup

1.  **Models**:
    - Place your YOLOv8 model (`.engine` or `.onnx`) in the `models/` directory.
    - **Crucial**: Update `nvdsinfer_config.txt` to point to your specific model file.
        ```ini
        model-engine-file=models/your_model.engine
        # or
        onnx-file=models/your_model.onnx
        ```

2.  **Config**:
    - Edit `config.yaml` to set your camera Device IDs (e.g., `Hikrobot-MV-CS023...`).

## Build & Run (Linux Only)

We have provided a convenience script for Linux systems.

1.  **Make script executable**:
    ```bash
    chmod +x run_jetson.sh
    ```

2.  **Run**:
    ```bash
    ./run_jetson.sh
    ```

### Manual Commands

**Build:**
```bash
sudo docker-compose build
```

**Run:**
```bash
sudo docker-compose up
```

## Accessing the Stream

Since the container runs in `host` network mode:
- Open your browser on the Jetson or a device on the same network.
- Go to: `http://<JETSON_IP>:8000`

## Troubleshooting on Jetson

- **"pyds not found"**: The Dockerfile compiles `pyds` bindings. Ensure the build completed successfully.
- **Camera not found**: Ensure you can see cameras with `arv-tool-0.8` on the host. The container uses `--network host` to access them.
- **Display issues**: If running headless, ignore X11 warnings. If using a monitor, ensure `xhost +` is run on the host.

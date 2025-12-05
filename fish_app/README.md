# Fish Monitoring System (DeepStream Version)

This application uses NVIDIA DeepStream to monitor fish behavior using two GigE cameras (Top and Side views).

## Prerequisites

- NVIDIA Jetson or x86 GPU with NVIDIA Drivers installed.
- Docker and NVIDIA Container Toolkit installed.
- Aravis-compatible GigE cameras connected.

## Setup

1.  **Models**: Place your YOLOv8 models (converted to engine or onnx if possible, or use DeepStream compatible weights) in the `models/` directory.
    - Update `config.yaml` with the correct model paths.
    - Update `nvdsinfer_config.txt` to point to your model engine/onnx file.

2.  **Configuration**:
    - Edit `config.yaml` to set camera IDs (`top_source`, `side_source`) and other parameters.

## Building

Build the Docker container:

```bash
docker-compose build
```

## Running

Run the application:

```bash
docker-compose up
```

The application will start the DeepStream pipeline and a web server.
Access the video stream at: `http://<device-ip>:8000`

## Troubleshooting

- **Camera Connection**: Ensure cameras are reachable via `arv-tool-0.8`.
- **DeepStream Errors**: Check `nvdsinfer_config.txt` paths.
- **Performance**: Adjust `fps` in `config.yaml` if bandwidth is saturated.

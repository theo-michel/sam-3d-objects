# SAM 3D Objects API Guide

## Prerequisites

Before running the server, ensure you have followed the [setup instructions](doc/setup.md) and have the following:

1.  **Dependencies**: Install all required packages.
    ```bash
    pip install -r requirements.txt
    ```
2.  **FAL API Key**: You need a key from [fal.ai](https://fal.ai) for the segmentation service.
    ```bash
    export FAL_KEY="your_fal_key_here"
    ```
3.  **Model Checkpoints**: Ensure your `checkpoints/` directory is populated.

## Running the Server

To start the server, run:

```bash
python api.py
```

The server will start listening on port `7310`.

## Exposing the API on a Deployed GPU

If you are running this on a remote GPU instance (e.g., AWS, GCP, Lambda Labs, RunPod), here are a few ways to access the API:

### Option 1: Direct Access (Public IP)

If your instance has a public IP and port 7310 is open in the firewall/security group:

1.  Ensure the server is running with `--host 0.0.0.0` (default in `api.py`).
2.  Access via: `http://<YOUR_INSTANCE_IP>:7310/docs`

### Option 2: SSH Tunneling (Secure & Recommended)

If you have SSH access to the machine, you can forward the port to your local machine without opening firewall ports.

On your **local machine**, run:

```bash
ssh -L 7310:localhost:7310 user@your-gpu-instance-ip
```

Now, you can access the API locally at: `http://localhost:7310/docs`

### Option 3: Using ngrok (Easy Public URL)

If you want a public URL (e.g., for testing with a frontend or mobile app) and don't want to mess with firewalls:

1.  Install ngrok on the GPU machine.
2.  Run the python server in one terminal.
3.  In another terminal, run:
    ```bash
    ngrok http 7310
    ```
4.  ngrok will give you a public URL (e.g., `https://random-name.ngrok-free.app`) that forwards to your local server.

## API Usage

### Endpoint: `/image-to-3d`

**Method**: `POST`

**Parameters**:
- `image`: The input image file.

**Response**:
- A ZIP file containing `.ply` files (Gaussian Splats) and a `transformations.json`.

**Example using cURL**:

```bash
curl -X POST "http://localhost:7310/image-to-3d" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/path/to/your/image.jpg" \
  --output results.zip
```

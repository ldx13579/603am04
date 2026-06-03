"""Cloud DQN Controller - Entry Point

启动方式:
    python run_api.py [--port 8000] [--model MODEL_PATH]

访问:
    http://localhost:8000/          前端仪表板
    http://localhost:8000/docs      API文档 (Swagger)
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="DQN Cloud Controller API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--model", default=None, help="Path to pre-trained model .pth file")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    if args.model:
        import os
        os.environ["DQN_MODEL_PATH"] = args.model

    print(f"Starting DQN Cloud Controller...")
    print(f"  Dashboard: http://localhost:{args.port}/")
    print(f"  API Docs:  http://localhost:{args.port}/docs")
    print(f"  WebSocket: ws://localhost:{args.port}/ws/realtime")

    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

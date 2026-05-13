"""ZMQ publisher for simulator state needed by task-space WBC inference."""

from __future__ import annotations

import json

import zmq
from loguru import logger


class WbcStatePub:
    """Publish simulator contact/root state for inference-side WBC torque control."""

    def __init__(self, port: int = 5556):
        self.port = port
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self.enabled = False

    def start(self) -> None:
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(f"tcp://*:{self.port}")
            self.enabled = True
            logger.info(f"WBC state publisher started on port {self.port}")
        except Exception as exc:
            logger.error(f"Failed to start WBC state publisher: {exc}")
            self.enabled = False

    def publish(self, payload: dict) -> None:
        if not self.enabled or self.socket is None:
            return
        try:
            self.socket.send_string(json.dumps(payload), zmq.NOBLOCK)
        except zmq.Again:
            pass
        except Exception as exc:
            logger.warning(f"WBC state publish failed: {exc}")

    def close(self) -> None:
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self.enabled = False

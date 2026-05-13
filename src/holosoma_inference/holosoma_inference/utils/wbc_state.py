"""ZMQ subscriber for simulator state needed by task-space WBC inference."""

from __future__ import annotations

import json

import zmq
from loguru import logger


class WbcStateSub:
    """Subscribe to simulator contact/root state for inference-side WBC torque control."""

    def __init__(self, port: int = 5556):
        self.port = port
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self.latest: dict | None = None

    def start(self) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{self.port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVTIMEO, 1)
        logger.info(f"WBC state subscriber started, connecting to port {self.port}")

    def get_latest(self) -> dict | None:
        if self.socket is None:
            return self.latest

        while True:
            try:
                self.latest = json.loads(self.socket.recv_string(zmq.NOBLOCK))
            except zmq.Again:
                break
        return self.latest

    def close(self) -> None:
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()

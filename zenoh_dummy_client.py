#!/usr/bin/env python3
"""
Dummy Zenoh client for testing Autoware-over-Zenoh connectivity.

This is a *test harness only* -- it does NOT create an MCP server and is not
used by cosmos. Its sole purpose is to verify that a Zenoh session opened on
this (remote / vast.ai) box can talk to a `zenoh-bridge-ros2dds` running on
your local computer.

    ROS 2  <-DDS->  zenoh-bridge-ros2dds (your laptop)  <-Zenoh/TCP->  this client

What it does:
    * Publishes dummy MRM requests on  /minerva/mrm/command  as plain UTF-8
      command strings (e.g. "emergency_stop"), cycling on a timer. The receiving
      bridge decodes these with bytes(payload).decode('utf-8').strip().lower().
    * Subscribes to camera frames on    /minerva/camera/front/image  and prints
      derived stream health (resolution, encoding, FPS, staleness).
    * Subscribes to camera status on    /minerva/camera/front/status  and prints
      the device-reported status. The status message type is unknown here, so the
      decoder tries std_msgs/String and falls back to raw bytes.

Topic mapping note:
    zenoh-bridge-ros2dds strips the leading '/' when mapping ROS topics onto
    Zenoh keys, so ROS '/minerva/mrm/command' becomes Zenoh 'minerva/mrm/command'.

Environment:
    ZENOH_CONNECT   Comma-separated Zenoh endpoints to connect to. Point this at
                    your local computer's public IP and the port its
                    zenoh-bridge listens on, e.g.
                        export ZENOH_CONNECT=tcp/203.0.113.7:7447
                    Defaults to 'tcp/localhost:7447'.
    MRM_PERIOD_S    Seconds between dummy MRM publishes (default 3.0).
    STATUS_PERIOD_S Seconds between camera-status prints    (default 2.0).
    MRM_COMMANDS    Comma-separated command strings to cycle through. Defaults to
                    'emergency_stop,comfortable_stop,pull_over,none'. Set this to
                    match whatever tokens your bridge accepts.

Usage:
    export ZENOH_CONNECT=tcp/<your-local-computer-ip>:7447
    python zenoh_dummy_client.py
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field

import zenoh
from pycdr2 import IdlStruct
from pycdr2.types import int32, sequence, uint8, uint32


# ---------------------------------------------------------------------------
# Zenoh keys (ROS '/foo/bar' -> Zenoh 'foo/bar' via zenoh-bridge-ros2dds)
# ---------------------------------------------------------------------------
KEY_MRM_COMMAND = "minerva/mrm/command"
KEY_CAMERA_IMAGE = "minerva/camera/front/image"
KEY_CAMERA_STATUS = "minerva/camera/front/status"


# ===================================================================
# CDR / IDL dataclasses -- serialized on the wire exactly like the
# corresponding ROS 2 messages so zenoh-bridge-ros2dds can relay them.
# ===================================================================
@dataclass
class Time(IdlStruct, typename="builtin_interfaces/Time"):
    sec: int32 = 0
    nanosec: uint32 = 0


@dataclass
class Header(IdlStruct, typename="std_msgs/Header"):
    stamp: Time = field(default_factory=Time)
    frame_id: str = ""


@dataclass
class Image(IdlStruct, typename="sensor_msgs/Image"):
    header: Header = field(default_factory=Header)
    height: uint32 = 0
    width: uint32 = 0
    encoding: str = ""
    is_bigendian: uint8 = 0
    step: uint32 = 0
    data: sequence[uint8] = field(default_factory=list)


@dataclass
class StringMsg(IdlStruct, typename="std_msgs/String"):
    data: str = ""


# ===================================================================
# Camera state tracker -- combines derived stream health (from the image
# topic) with the device-reported status (from the status topic).
# ===================================================================
class CameraMonitor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_msg: Image | None = None
        self._last_rx_time: float | None = None
        self._frame_count = 0
        # Timestamps of recent frames, for a rolling FPS estimate.
        self._recent: list[float] = []
        # Device-reported status (from /minerva/camera/front/status).
        self._last_status: str | None = None
        self._last_status_rx: float | None = None
        self._status_count = 0

    def on_image(self, sample: "zenoh.Sample") -> None:
        try:
            msg = Image.deserialize(bytes(sample.payload))
        except Exception as e:  # noqa: BLE001 - just diagnostics
            print(f"[camera] failed to deserialize frame: {e}", flush=True)
            return
        now = time.time()
        with self._lock:
            self._last_msg = msg
            self._last_rx_time = now
            self._frame_count += 1
            self._recent.append(now)
            # keep only the last ~5 seconds of timestamps
            cutoff = now - 5.0
            self._recent = [t for t in self._recent if t >= cutoff]

    def on_status(self, sample: "zenoh.Sample") -> None:
        payload = bytes(sample.payload)
        # Message type on /status is unknown -- try std_msgs/String, else raw.
        try:
            text = StringMsg.deserialize(payload).data
        except Exception:  # noqa: BLE001 - unknown/other message type
            text = f"<{len(payload)} raw bytes: {payload[:32].hex()}...>"
        with self._lock:
            self._last_status = text
            self._last_status_rx = time.time()
            self._status_count += 1

    def status(self) -> dict:
        with self._lock:
            msg = self._last_msg
            last_rx = self._last_rx_time
            count = self._frame_count
            recent = list(self._recent)
            status_text = self._last_status
            status_rx = self._last_status_rx
            status_count = self._status_count

        reported = {
            "reported_status": status_text,
            "reported_status_count": status_count,
            "reported_status_age_s": (
                round(time.time() - status_rx, 3) if status_rx is not None else None
            ),
        }

        if msg is None or last_rx is None:
            return {
                "online": False,
                "frames_received": 0,
                "detail": "no frames received yet",
                **reported,
            }

        age = time.time() - last_rx
        fps = 0.0
        if len(recent) >= 2:
            span = recent[-1] - recent[0]
            if span > 0:
                fps = (len(recent) - 1) / span

        return {
            "online": age < 2.0,
            "frames_received": count,
            "width": int(msg.width),
            "height": int(msg.height),
            "encoding": msg.encoding,
            "bytes_per_frame": len(msg.data),
            "fps_est": round(fps, 2),
            "last_frame_age_s": round(age, 3),
            "frame_id": msg.header.frame_id,
            **reported,
        }


# ===================================================================
# Zenoh client
# ===================================================================
class DummyClient:
    def __init__(self) -> None:
        endpoints = os.environ.get("ZENOH_CONNECT", "tcp/localhost:7447")
        endpoint_list = [e.strip() for e in endpoints.split(",") if e.strip()]

        cfg = zenoh.Config()
        cfg.insert_json5("mode", '"client"')
        cfg.insert_json5("connect/endpoints", json.dumps(endpoint_list))

        print(f"[zenoh] connecting to {endpoint_list} ...", flush=True)
        self._session = zenoh.open(cfg)
        print("[zenoh] session open", flush=True)

        self.pub_mrm = self._session.declare_publisher(KEY_MRM_COMMAND)

        self.camera = CameraMonitor()
        self._sub_camera = self._session.declare_subscriber(
            KEY_CAMERA_IMAGE, self.camera.on_image
        )
        print(f"[zenoh] subscribed to '{KEY_CAMERA_IMAGE}'", flush=True)

        self._sub_status = self._session.declare_subscriber(
            KEY_CAMERA_STATUS, self.camera.on_status
        )
        print(f"[zenoh] subscribed to '{KEY_CAMERA_STATUS}'", flush=True)

    def publish_mrm(self, command: str) -> None:
        # The bridge decodes the payload as a UTF-8 string:
        #   command = bytes(sample.payload).decode('utf-8').strip().lower()
        # so we publish a plain lowercase string, NOT a CDR-serialized struct.
        self.pub_mrm.put(command.encode("utf-8"))
        print(
            f"[mrm] published '{KEY_MRM_COMMAND}' command={command!r}",
            flush=True,
        )

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass


# ===================================================================
# Entry point -- run publish/status loops until Ctrl-C.
# ===================================================================
def main() -> None:
    mrm_period = float(os.environ.get("MRM_PERIOD_S", "3.0"))
    status_period = float(os.environ.get("STATUS_PERIOD_S", "2.0"))

    client = DummyClient()

    # Cycle through the dummy MRM command strings so the bridge sees variety.
    # Override with MRM_COMMANDS="foo,bar,..." to match your bridge's vocabulary.
    commands_env = os.environ.get(
        "MRM_COMMANDS", "emergency_stop,comfortable_stop,pull_over,cancel_all,normal"
    )
    command_cycle = [c.strip() for c in commands_env.split(",") if c.strip()]

    next_mrm = time.time()
    next_status = time.time()
    idx = 0

    print(
        f"[run] mrm every {mrm_period}s, camera status every {status_period}s "
        "(Ctrl-C to stop)",
        flush=True,
    )
    try:
        while True:
            now = time.time()

            if now >= next_mrm:
                command = command_cycle[idx % len(command_cycle)]
                try:
                    client.publish_mrm(command)
                except Exception as e:  # noqa: BLE001
                    print(f"[mrm] publish failed: {e}", flush=True)
                idx += 1
                next_mrm = now + mrm_period

            if now >= next_status:
                status = client.camera.status()
                print(f"[camera] status: {json.dumps(status)}", flush=True)
                next_status = now + status_period

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[run] stopping", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MCP Server exposing three Autoware Minimal Risk Maneuvers (MRMs) over Zenoh.

Unlike autoware_mcp_server.py (which uses native rclpy), this server talks to
Autoware remotely via a `zenoh-bridge-ros2dds` instance that runs on the
Autoware/CARLA host. The VLM and this MCP server can therefore run on a
GPU box (e.g. vast.ai) that does not have ROS 2 installed.

Wire format:
    ROS 2  <-DDS->  zenoh-bridge-ros2dds  <-Zenoh/TCP->  this server

Tools exposed:
    - emergency_stop      -> behavior = EMERGENCY_STOP (1)
    - comfortable_stop    -> behavior = COMFORTABLE_STOP (2)
    - pull_over           -> behavior = PULL_OVER (3)
    - clear_mrm           -> behavior = NONE (0), state = NORMAL
    - get_vehicle_state   -> cached velocity + steering telemetry

Environment:
    ZENOH_CONNECT  Comma-separated Zenoh endpoints to connect to, e.g.
                   'tcp/203.0.113.5:7447'. Defaults to 'tcp/localhost:7447'.

Usage (standalone test):
    export ZENOH_CONNECT=tcp/<autoware-host>:7447
    python zenoh_autoware_mcp_server.py

Usage (spawned over stdio by multiview_nobatch_fp16_inference.py):
    Automatically managed -- do not run manually in that case.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import zenoh
from pycdr2 import IdlStruct
from pycdr2.types import float32, int32, uint16, uint32

from fastmcp import FastMCP


# ---------------------------------------------------------------------------
# Topic names (zenoh-bridge-ros2dds maps ROS '/foo/bar' -> Zenoh 'foo/bar')
# ---------------------------------------------------------------------------
KEY_MRM_REQUEST = "system/mrm_request"
KEY_VELOCITY_STATUS = "vehicle/status/velocity_status"
KEY_STEERING_STATUS = "vehicle/status/steering_status"


# ---------------------------------------------------------------------------
# MRM behavior / state enums (mirroring tier4_system_msgs/MrmBehaviorStatus)
# ---------------------------------------------------------------------------
class MrmState:
    NOT_AVAILABLE = 0
    NORMAL = 1
    OPERATING = 2
    SUCCEEDED = 3
    FAILED = 4


class MrmBehavior:
    NONE = 0
    EMERGENCY_STOP = 1
    COMFORTABLE_STOP = 2
    PULL_OVER = 3


# ===================================================================
# CDR / IDL dataclasses -- serialized on the wire exactly like the
# corresponding ROS 2 messages, so the zenoh-bridge-ros2dds can relay
# them to the matching DDS topics on the Autoware side.
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
class MrmBehaviorStatusMsg(IdlStruct, typename="tier4_system_msgs/MrmBehaviorStatus"):
    stamp: Header = field(default_factory=Header)
    state: uint16 = 0
    behavior: uint16 = 0


@dataclass
class VelocityReportMsg(IdlStruct, typename="autoware_auto_vehicle_msgs/VelocityReport"):
    header: Header = field(default_factory=Header)
    longitudinal_velocity: float32 = 0.0
    lateral_velocity: float32 = 0.0
    heading_rate: float32 = 0.0


@dataclass
class SteeringReportMsg(IdlStruct, typename="autoware_auto_vehicle_msgs/SteeringReport"):
    stamp: Time = field(default_factory=Time)
    steering_tire_angle: float32 = 0.0


# ===================================================================
# Zenoh bridge (lazy session, publishers, cached subscribers)
# ===================================================================
class ZenohBridge:
    """Thin wrapper owning the Zenoh session and all publishers/subscribers."""

    def __init__(self) -> None:
        endpoints = os.environ.get("ZENOH_CONNECT", "tcp/localhost:7447")
        endpoint_list = [e.strip() for e in endpoints.split(",") if e.strip()]

        cfg = zenoh.Config()
        cfg.insert_json5("mode", '"client"')
        cfg.insert_json5(
            "connect/endpoints", json.dumps(endpoint_list)
        )

        self._session = zenoh.open(cfg)

        self.pub_mrm_request = self._session.declare_publisher(KEY_MRM_REQUEST)

        self._velocity: VelocityReportMsg | None = None
        self._steering: SteeringReportMsg | None = None
        self._lock = threading.Lock()

        self._sub_velocity = self._session.declare_subscriber(
            KEY_VELOCITY_STATUS, self._on_velocity
        )
        self._sub_steering = self._session.declare_subscriber(
            KEY_STEERING_STATUS, self._on_steering
        )

    # -- subscription callbacks ----------------------------------------
    def _on_velocity(self, sample: "zenoh.Sample") -> None:
        try:
            payload = bytes(sample.payload)
            msg = VelocityReportMsg.deserialize(payload)
        except Exception:
            return
        with self._lock:
            self._velocity = msg

    def _on_steering(self, sample: "zenoh.Sample") -> None:
        try:
            payload = bytes(sample.payload)
            msg = SteeringReportMsg.deserialize(payload)
        except Exception:
            return
        with self._lock:
            self._steering = msg

    # -- publish helpers -----------------------------------------------
    def publish_mrm(self, behavior: int, state: int = MrmState.OPERATING) -> None:
        now = time.time()
        sec = int(now)
        nanosec = int((now - sec) * 1e9)
        msg = MrmBehaviorStatusMsg(
            stamp=Header(stamp=Time(sec=sec, nanosec=nanosec), frame_id=""),
            state=state,
            behavior=behavior,
        )
        self.pub_mrm_request.put(msg.serialize())

    # -- state readout -------------------------------------------------
    def get_cached_state(self) -> dict[str, Any]:
        with self._lock:
            vr = self._velocity
            sr = self._steering

        state: dict[str, Any] = {}
        if vr is not None:
            lon = float(vr.longitudinal_velocity)
            lat = float(vr.lateral_velocity)
            state["velocity_mps"] = round(math.sqrt(lon * lon + lat * lat), 3)
            state["longitudinal_velocity_mps"] = round(lon, 3)
            state["lateral_velocity_mps"] = round(lat, 3)
            state["heading_rate_rps"] = round(float(vr.heading_rate), 3)
        else:
            state["velocity_mps"] = None

        if sr is not None:
            ang_rad = float(sr.steering_tire_angle)
            state["steering_tire_angle_rad"] = round(ang_rad, 4)
            state["steering_tire_angle_deg"] = round(math.degrees(ang_rad), 2)
        else:
            state["steering_tire_angle_deg"] = None
        return state

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass


# ===================================================================
# Module-level lazy singleton
# ===================================================================
_bridge: ZenohBridge | None = None
_bridge_lock = threading.Lock()


def _start_zenoh() -> ZenohBridge:
    global _bridge
    with _bridge_lock:
        if _bridge is None:
            _bridge = ZenohBridge()
    return _bridge


def _ok(message: str) -> str:
    return json.dumps({"success": True, "message": message})


def _err(message: str) -> str:
    return json.dumps({"success": False, "message": message})


def _publish_mrm(behavior: int, label: str) -> str:
    try:
        bridge = _start_zenoh()
        bridge.publish_mrm(behavior, state=MrmState.OPERATING)
    except Exception as e:
        return _err(f"Zenoh publish failed for {label}: {e}")
    return _ok(f"{label} requested via Zenoh (behavior={behavior})")


# ===================================================================
# MCP Server
# ===================================================================
mcp = FastMCP("autoware-zenoh-mrm")


@mcp.tool()
def emergency_stop() -> str:
    """Trigger the EMERGENCY_STOP Minimal Risk Maneuver.

    Publishes tier4_system_msgs/MrmBehaviorStatus on /system/mrm_request with
    behavior=EMERGENCY_STOP. Autoware will hard-brake the ego vehicle.
    Use clear_mrm() to resume afterwards.
    """
    return _publish_mrm(MrmBehavior.EMERGENCY_STOP, "EMERGENCY_STOP")


@mcp.tool()
def comfortable_stop() -> str:
    """Trigger the COMFORTABLE_STOP Minimal Risk Maneuver.

    Publishes tier4_system_msgs/MrmBehaviorStatus on /system/mrm_request with
    behavior=COMFORTABLE_STOP. Autoware will decelerate smoothly to a stop.
    Use clear_mrm() to resume afterwards.
    """
    return _publish_mrm(MrmBehavior.COMFORTABLE_STOP, "COMFORTABLE_STOP")


@mcp.tool()
def pull_over() -> str:
    """Trigger the PULL_OVER Minimal Risk Maneuver.

    Publishes tier4_system_msgs/MrmBehaviorStatus on /system/mrm_request with
    behavior=PULL_OVER. Autoware will leave the travel lane and stop at the
    road shoulder. Use clear_mrm() to resume afterwards.
    """
    return _publish_mrm(MrmBehavior.PULL_OVER, "PULL_OVER")


@mcp.tool()
def clear_mrm() -> str:
    """Clear the current Minimal Risk Maneuver and restore normal operation.

    Publishes behavior=NONE, state=NORMAL on /system/mrm_request. This is the
    Zenoh equivalent of calling the /system/clear_emergency service.
    """
    try:
        bridge = _start_zenoh()
        bridge.publish_mrm(MrmBehavior.NONE, state=MrmState.NORMAL)
    except Exception as e:
        return _err(f"Zenoh publish failed for clear_mrm: {e}")
    return _ok("MRM cleared (behavior=NONE, state=NORMAL)")


@mcp.tool()
def get_vehicle_state() -> str:
    """Return the latest cached vehicle state (speed, steering angle).

    Read-only: subscribes to /vehicle/status/velocity_status and
    /vehicle/status/steering_status via Zenoh and returns the most recent
    sample. Fields are null until the first message arrives.
    """
    try:
        bridge = _start_zenoh()
    except Exception as e:
        return _err(f"Zenoh session unavailable: {e}")
    return json.dumps(bridge.get_cached_state())


# ===================================================================
# Entry point
# ===================================================================
if __name__ == "__main__":
    _start_zenoh()
    mcp.run()

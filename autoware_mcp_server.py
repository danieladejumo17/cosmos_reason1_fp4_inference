#!/usr/bin/env python3
"""
MCP Server exposing Autoware vehicle control tools over native ROS 2 (rclpy).

Runs as a stdio MCP server. A background thread spins the rclpy executor so
that ROS 2 subscriptions stay alive while the async MCP event loop handles
tool calls.

Usage (standalone test):
    python autoware_mcp_server.py

Usage (spawned by vlm_autoware_loop.py via MCP stdio transport):
    Automatically managed — do not run manually in that case.
"""

from __future__ import annotations

import json
import math
import threading
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_control_msgs.msg import AckermannLateralCommand
from autoware_auto_control_msgs.msg import LongitudinalCommand
from autoware_auto_vehicle_msgs.msg import VelocityReport, SteeringReport
from tier4_planning_msgs.msg import VelocityLimit
from tier4_system_msgs.msg import MrmBehaviorStatus
from std_srvs.srv import Trigger

from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_VELOCITY_MPS = 30.0
MIN_VELOCITY_MPS = 0.0
MAX_STEER_DEG = 40.0
DEFAULT_ESTOP_DECEL = -2.5  # m/s²
DEFAULT_ESTOP_JERK = -1.5   # m/s³

CTRL_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
STATUS_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


# ===================================================================
# ROS 2 bridge node
# ===================================================================
class AutowareBridge(Node):
    """Thin wrapper that owns all Autoware publishers / subscribers."""

    def __init__(self) -> None:
        super().__init__("vlm_mcp_bridge")

        # Publishers
        self.pub_velocity_limit = self.create_publisher(
            VelocityLimit,
            "/planning/scenario_planning/max_velocity_default",
            CTRL_QOS,
        )
        self.pub_external_cmd = self.create_publisher(
            AckermannControlCommand,
            "/external/selected/control_cmd",
            CTRL_QOS,
        )
        self.pub_mrm_operate = self.create_publisher(
            MrmBehaviorStatus,
            "/system/mrm/emergency_stop/operate",
            CTRL_QOS,
        )

        # Service clients
        self.cli_clear_emergency = self.create_client(
            Trigger, "/system/clear_emergency"
        )

        # Subscriptions (cached state)
        self._velocity_report: VelocityReport | None = None
        self._steering_report: SteeringReport | None = None

        self.create_subscription(
            VelocityReport,
            "/vehicle/status/velocity_status",
            self._on_velocity,
            STATUS_QOS,
        )
        self.create_subscription(
            SteeringReport,
            "/vehicle/status/steering_status",
            self._on_steering,
            STATUS_QOS,
        )

    # -- subscription callbacks ------------------------------------------
    def _on_velocity(self, msg: VelocityReport) -> None:
        self._velocity_report = msg

    def _on_steering(self, msg: SteeringReport) -> None:
        self._steering_report = msg

    # -- helpers ---------------------------------------------------------
    def get_cached_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        vr = self._velocity_report
        if vr is not None:
            state["velocity_mps"] = round(
                math.sqrt(
                    vr.longitudinal_velocity ** 2
                    + vr.lateral_velocity ** 2
                ), 3,
            )
            state["longitudinal_velocity_mps"] = round(vr.longitudinal_velocity, 3)
            state["lateral_velocity_mps"] = round(vr.lateral_velocity, 3)
            state["heading_rate_rps"] = round(vr.heading_rate, 3)
        else:
            state["velocity_mps"] = None

        sr = self._steering_report
        if sr is not None:
            state["steering_tire_angle_rad"] = round(sr.steering_tire_angle, 4)
            state["steering_tire_angle_deg"] = round(
                math.degrees(sr.steering_tire_angle), 2
            )
        else:
            state["steering_tire_angle_deg"] = None
        return state


# ===================================================================
# Background ROS 2 spinner
# ===================================================================
_bridge: AutowareBridge | None = None
_spin_thread: threading.Thread | None = None


def _start_ros2() -> AutowareBridge:
    global _bridge, _spin_thread
    if _bridge is not None:
        return _bridge

    rclpy.init()
    _bridge = AutowareBridge()

    def _spin() -> None:
        assert _bridge is not None
        rclpy.spin(_bridge)

    _spin_thread = threading.Thread(target=_spin, daemon=True)
    _spin_thread.start()
    _bridge.get_logger().info("AutowareBridge ROS 2 node started")
    return _bridge


def _ok(message: str) -> str:
    return json.dumps({"success": True, "message": message})


def _err(message: str) -> str:
    return json.dumps({"success": False, "message": message})


# ===================================================================
# MCP Server
# ===================================================================
mcp = FastMCP("autoware-control")


@mcp.tool()
def set_velocity(velocity_mps: float) -> str:
    """Set the external velocity limit for the ego vehicle (m/s).

    Autoware's planner will respect this as an upper bound.
    Clamped to [0, 30] m/s.
    """
    bridge = _start_ros2()
    velocity_mps = max(MIN_VELOCITY_MPS, min(MAX_VELOCITY_MPS, velocity_mps))

    msg = VelocityLimit()
    msg.max_velocity = velocity_mps
    bridge.pub_velocity_limit.publish(msg)
    bridge.get_logger().info(f"set_velocity: {velocity_mps:.2f} m/s")
    return _ok(f"Velocity limit set to {velocity_mps:.2f} m/s")


@mcp.tool()
def emergency_stop() -> str:
    """Trigger an immediate emergency stop (Minimal Risk Maneuver).

    The vehicle will decelerate at ~2.5 m/s² until stopped.
    Use clear_emergency() to resume afterwards.
    """
    bridge = _start_ros2()

    msg = MrmBehaviorStatus()
    msg.state = MrmBehaviorStatus.OPERATING
    bridge.pub_mrm_operate.publish(msg)

    cmd = AckermannControlCommand()
    cmd.longitudinal.speed = 0.0
    cmd.longitudinal.acceleration = DEFAULT_ESTOP_DECEL
    cmd.longitudinal.jerk = DEFAULT_ESTOP_JERK
    bridge.pub_external_cmd.publish(cmd)

    bridge.get_logger().warn("EMERGENCY STOP triggered")
    return _ok("Emergency stop triggered")


@mcp.tool()
def clear_emergency() -> str:
    """Clear the emergency state so the vehicle can resume driving.

    Calls the /system/clear_emergency service.
    """
    bridge = _start_ros2()

    if not bridge.cli_clear_emergency.wait_for_service(timeout_sec=2.0):
        return _err("clear_emergency service not available")

    future = bridge.cli_clear_emergency.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(bridge, future, timeout_sec=2.0)

    if future.result() is not None:
        resp = future.result()
        bridge.get_logger().info(f"clear_emergency: success={resp.success}")
        if resp.success:
            return _ok("Emergency state cleared")
        return _err(f"Service returned failure: {resp.message}")
    return _err("Service call timed out")


@mcp.tool()
def stop_override(deceleration_mps2: float = 2.0) -> str:
    """Command a controlled stop with a specified deceleration (m/s²).

    Unlike emergency_stop, this uses the external control path through
    vehicle_cmd_gate and allows configurable deceleration.
    """
    bridge = _start_ros2()
    decel = -abs(deceleration_mps2)

    cmd = AckermannControlCommand()
    cmd.longitudinal.speed = 0.0
    cmd.longitudinal.acceleration = float(decel)
    cmd.longitudinal.jerk = float(decel * 0.6)
    cmd.lateral.steering_tire_angle = 0.0
    bridge.pub_external_cmd.publish(cmd)

    bridge.get_logger().info(f"stop_override: decel={decel:.2f} m/s²")
    return _ok(f"Stop override sent (decel={decel:.2f} m/s²)")


@mcp.tool()
def steer(direction: str, angle_deg: float) -> str:
    """Publish a steering command (left or right) through the external control path.

    Args:
        direction: "left" or "right"
        angle_deg: Steering angle in degrees (0-40).
    """
    bridge = _start_ros2()
    direction = direction.strip().lower()
    if direction not in ("left", "right"):
        return _err(f"Invalid direction '{direction}'. Use 'left' or 'right'.")

    angle_deg = max(0.0, min(MAX_STEER_DEG, abs(angle_deg)))
    angle_rad = math.radians(angle_deg)
    if direction == "right":
        angle_rad = -angle_rad

    cmd = AckermannControlCommand()
    cmd.lateral.steering_tire_angle = float(angle_rad)
    cmd.lateral.steering_tire_rotation_rate = 0.5
    bridge.pub_external_cmd.publish(cmd)

    bridge.get_logger().info(
        f"steer: {direction} {angle_deg:.1f}° ({angle_rad:.4f} rad)"
    )
    return _ok(f"Steering {direction} {angle_deg:.1f}°")


@mcp.tool()
def lane_switch(direction: str) -> str:
    """Request a lane change (left or right).

    Publishes a velocity-limit nudge and logs the request. The Autoware
    behavior planner will execute the actual lane change maneuver.

    Args:
        direction: "left" or "right"
    """
    bridge = _start_ros2()
    direction = direction.strip().lower()
    if direction not in ("left", "right"):
        return _err(f"Invalid direction '{direction}'. Use 'left' or 'right'.")

    angle_deg = 5.0 if direction == "left" else -5.0
    angle_rad = math.radians(angle_deg)

    cmd = AckermannControlCommand()
    cmd.lateral.steering_tire_angle = float(angle_rad)
    cmd.lateral.steering_tire_rotation_rate = 0.3
    bridge.pub_external_cmd.publish(cmd)

    bridge.get_logger().info(f"lane_switch: {direction}")
    return _ok(f"Lane switch {direction} requested")


@mcp.tool()
def get_vehicle_state() -> str:
    """Return the latest cached vehicle state (speed, steering angle).

    This is a read-only tool — it does not send any commands.
    """
    bridge = _start_ros2()
    state = bridge.get_cached_state()
    return json.dumps(state)


# ===================================================================
# Entry point
# ===================================================================
if __name__ == "__main__":
    _start_ros2()
    mcp.run()

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Isaac Sim Extension: JetBot Sphere-Following Policy Playback
#
# Provides a UI panel to load the trained PPO checkpoint and watch the JetBot
# chase a green sphere in real-time inside Isaac Sim.

import asyncio
import math
import os

import numpy as np
import omni.ext
import omni.kit.app
import omni.ui as ui
import omni.usd
import torch
import torch.nn as nn

from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade


# ==============================================================================
# Quaternion utilities (wxyz convention, matching Isaac Lab math_utils)
# ==============================================================================

def quat_rotate(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vector by quaternion (wxyz)."""
    xyz = quat_wxyz[1:4]
    t = 2.0 * np.cross(xyz, vec)
    return vec + quat_wxyz[0] * t + np.cross(xyz, t)


def quat_rotate_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vector by inverse quaternion (wxyz). World -> body frame."""
    xyz = quat_wxyz[1:4]
    t = 2.0 * np.cross(xyz, vec)
    return vec - quat_wxyz[0] * t + np.cross(xyz, t)


# ==============================================================================
# Policy network (matches skrl training architecture)
# ==============================================================================

class PolicyNetwork(nn.Module):
    """Policy: 4 obs -> [64, ELU, 64, ELU] -> 2 actions."""

    def __init__(self):
        super().__init__()
        self.net_container = nn.Sequential(
            nn.Linear(4, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
        )
        self.policy_layer = nn.Linear(64, 2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy_layer(self.net_container(obs))


def load_policy(checkpoint_path: str):
    """Load policy weights and observation normalizer from skrl checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy = PolicyNetwork()
    state_dict = {
        key: ckpt["policy"][key]
        for key in [
            "net_container.0.weight", "net_container.0.bias",
            "net_container.2.weight", "net_container.2.bias",
            "policy_layer.weight", "policy_layer.bias",
        ]
    }
    policy.load_state_dict(state_dict)
    policy.eval()

    obs_mean = ckpt["state_preprocessor"]["running_mean"].float()
    obs_var = ckpt["state_preprocessor"]["running_variance"].float()
    obs_std = torch.sqrt(obs_var + 1e-8)
    return policy, obs_mean, obs_std


# ==============================================================================
# Observation computation (matches SphereFollowEnv._get_observations exactly)
# ==============================================================================

FORWARD_VEC_B = np.array([1.0, 0.0, 0.0])


def compute_observations(
    robot_pos: np.ndarray,
    robot_quat_wxyz: np.ndarray,
    robot_lin_vel_w: np.ndarray,
    sphere_pos: np.ndarray,
    max_distance: float = 3.0,
) -> np.ndarray:
    """Compute 4D observation: [dot, cross_z, dist_norm, forward_speed]."""
    forward_w = quat_rotate(robot_quat_wxyz, FORWARD_VEC_B)

    dir_to_sphere = sphere_pos - robot_pos
    dir_to_sphere[2] = 0.0
    dist = max(np.linalg.norm(dir_to_sphere[:2]), 1e-6)
    dir_norm = dir_to_sphere / dist

    dot = np.dot(forward_w, dir_norm)
    cross_z = forward_w[0] * dir_norm[1] - forward_w[1] * dir_norm[0]
    dist_norm = min(dist / max_distance, 1.0)

    lin_vel_b = quat_rotate_inverse(robot_quat_wxyz, robot_lin_vel_w)
    forward_speed = lin_vel_b[0]

    return np.array([dot, cross_z, dist_norm, forward_speed], dtype=np.float32)


# ==============================================================================
# Scene helpers
# ==============================================================================

def create_green_sphere(stage, position: np.ndarray, radius: float = 0.1):
    """Create a green sphere visual prim."""
    sphere_path = "/World/TargetSphere"
    sphere_prim = UsdGeom.Sphere.Define(stage, sphere_path)
    sphere_prim.GetRadiusAttr().Set(radius)
    sphere_prim.AddTranslateOp().Set(Gf.Vec3f(*position.tolist()))

    mat_path = "/World/TargetSphere/GreenMaterial"
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 1.0, 0.0))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(sphere_prim.GetPrim()).Bind(material)
    return sphere_prim


def set_sphere_position(sphere_prim, position: np.ndarray):
    """Update sphere translate op."""
    for op in sphere_prim.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3f(*position.tolist()))
            return


def create_dome_light(stage, intensity: float = 2000.0):
    """Create dome light matching training environment."""
    light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    light.CreateIntensityAttr().Set(intensity)
    light.CreateColorAttr().Set(Gf.Vec3f(0.75, 0.75, 0.75))


def random_sphere_position(robot_pos: np.ndarray, spawn_min=0.5, spawn_max=1.5, z=0.1) -> np.ndarray:
    """Random position at [spawn_min, spawn_max] distance from robot."""
    angle = np.random.uniform(0, 2 * math.pi)
    dist = np.random.uniform(spawn_min, spawn_max)
    return np.array([
        robot_pos[0] + dist * math.cos(angle),
        robot_pos[1] + dist * math.sin(angle),
        z,
    ])


# ==============================================================================
# Extension
# ==============================================================================

class SphereFollowExtension(omni.ext.IExt):
    """Isaac Sim extension for JetBot sphere-following policy playback."""

    def on_startup(self, ext_id):
        print("[SphereFollow] Extension startup")

        # State
        self._world = None
        self._jetbot = None
        self._sphere_prim = None
        self._policy = None
        self._obs_mean = None
        self._obs_std = None
        self._sphere_pos = np.array([0.8, 0.3, 0.1])
        self._spheres_reached = 0
        self._step_count = 0
        self._is_running = False
        self._left_wheel_idx = 0
        self._right_wheel_idx = 1
        self._decimation_counter = 0

        # Config (matching SphereFollowEnvCfg)
        self._sphere_reach_threshold = 0.3
        self._sphere_spawn_min = 0.5
        self._sphere_spawn_max = 1.5
        self._sphere_radius = 0.1
        self._max_distance = 3.0

        # Default checkpoint path
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_path = ext_manager.get_extension_path(ext_id)
        project_root = os.path.normpath(os.path.join(ext_path, "..", ".."))
        self._default_checkpoint = os.path.join(
            project_root,
            "logs", "skrl", "sphere_follow_direct",
            "2026-02-21_19-46-44_ppo_torch", "checkpoints", "best_agent.pt",
        )

        self._build_ui()

    def _build_ui(self):
        self._window = ui.Window("JetBot Sphere Follow", width=450, height=320)
        with self._window.frame:
            with ui.VStack(spacing=6, style={"margin": 8}):
                # Title
                ui.Label(
                    "JetBot Sphere-Following Policy Playback",
                    style={"font_size": 16, "color": 0xFFDDDDDD},
                    height=28,
                    alignment=ui.Alignment.CENTER,
                )

                ui.Line(height=2, style={"color": 0xFF444444})

                # Checkpoint path
                ui.Label("Checkpoint Path:", height=18)
                self._checkpoint_field = ui.StringField(height=22)
                self._checkpoint_field.model.set_value(self._default_checkpoint)

                ui.Spacer(height=6)

                # Control buttons
                with ui.HStack(height=32, spacing=6):
                    ui.Button(
                        "Load World",
                        clicked_fn=self._on_load,
                        style={"Button": {"background_color": 0xFF2D8CFF}},
                    )
                    ui.Button(
                        "Start",
                        clicked_fn=self._on_start,
                        style={"Button": {"background_color": 0xFF00CC66}},
                    )
                    ui.Button(
                        "Stop",
                        clicked_fn=self._on_stop,
                        style={"Button": {"background_color": 0xFFFF4444}},
                    )
                    ui.Button(
                        "Reset",
                        clicked_fn=self._on_reset,
                        style={"Button": {"background_color": 0xFFFFAA00}},
                    )

                ui.Spacer(height=6)
                ui.Line(height=2, style={"color": 0xFF444444})

                # Status
                self._status_label = ui.Label(
                    "Status: Ready - Press 'Load World' to begin",
                    height=20,
                    style={"color": 0xFFAAFFAA},
                )
                self._stats_label = ui.Label(
                    "Spheres reached: 0  |  Steps: 0",
                    height=20,
                )
                self._obs_label = ui.Label(
                    "Obs: [-, -, -, -]  Act: [-, -]",
                    height=20,
                    style={"color": 0xFFAAAAFF},
                )

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _on_load(self):
        self._status_label.text = "Status: Loading world..."
        asyncio.ensure_future(self._setup_world_async())

    def _on_start(self):
        if self._world is None:
            self._status_label.text = "Status: Load world first!"
            return
        if self._policy is None:
            self._status_label.text = "Status: No policy loaded!"
            return
        self._is_running = True
        self._world.play()
        self._status_label.text = "Status: Running trained policy..."

    def _on_stop(self):
        self._is_running = False
        if self._world is not None:
            self._world.pause()
        self._status_label.text = "Status: Paused"

    def _on_reset(self):
        if self._world is None:
            self._status_label.text = "Status: Load world first!"
            return
        self._is_running = False
        self._spheres_reached = 0
        self._step_count = 0
        self._decimation_counter = 0
        asyncio.ensure_future(self._reset_async())

    # ------------------------------------------------------------------
    # Async world setup
    # ------------------------------------------------------------------

    async def _setup_world_async(self):
        try:
            # Load checkpoint
            checkpoint_path = self._checkpoint_field.model.get_value_as_string()
            if not os.path.exists(checkpoint_path):
                self._status_label.text = f"Status: ERROR - checkpoint not found"
                print(f"[SphereFollow] Checkpoint not found: {checkpoint_path}")
                return

            self._policy, self._obs_mean, self._obs_std = load_policy(checkpoint_path)
            print(f"[SphereFollow] Policy loaded from {checkpoint_path}")

            # Create world (matching training: 120Hz physics, 60Hz render)
            self._world = World(
                stage_units_in_meters=1.0,
                physics_dt=1.0 / 120.0,
                rendering_dt=1.0 / 60.0,
            )
            await self._world.initialize_simulation_context_async()

            # Spawn JetBot
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                self._status_label.text = "Status: ERROR - Could not find assets"
                return

            jetbot_asset_path = assets_root_path + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
            self._jetbot = self._world.scene.add(
                WheeledRobot(
                    prim_path="/World/Jetbot",
                    name="sphere_follow_jetbot",
                    wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                    create_robot=True,
                    usd_path=jetbot_asset_path,
                    position=np.array([0, 0.0, 0.05]),
                )
            )

            # Ground plane
            self._world.scene.add_default_ground_plane()

            # Dome light and green sphere
            stage = omni.usd.get_context().get_stage()
            create_dome_light(stage, intensity=2000.0)

            self._sphere_pos = np.array([0.8, 0.3, 0.1])
            self._sphere_prim = create_green_sphere(stage, self._sphere_pos, radius=self._sphere_radius)

            # Reset world to initialize physics
            await self._world.reset_async()

            # Get wheel joint indices
            self._left_wheel_idx = self._jetbot.dof_names.index("left_wheel_joint")
            self._right_wheel_idx = self._jetbot.dof_names.index("right_wheel_joint")

            # Register physics callback
            self._world.add_physics_callback("sphere_follow_step", self._on_physics_step)

            self._spheres_reached = 0
            self._step_count = 0
            self._decimation_counter = 0
            self._is_running = False

            self._status_label.text = "Status: World loaded - Press 'Start'"
            print("[SphereFollow] World loaded successfully")

        except Exception as e:
            self._status_label.text = f"Status: ERROR - {e}"
            print(f"[SphereFollow] Setup error: {e}")
            import traceback
            traceback.print_exc()

    async def _reset_async(self):
        try:
            await self._world.reset_async()
            robot_pos, _ = self._jetbot.get_world_pose()
            self._sphere_pos = random_sphere_position(
                robot_pos, self._sphere_spawn_min, self._sphere_spawn_max, self._sphere_radius
            )
            set_sphere_position(self._sphere_prim, self._sphere_pos)
            self._stats_label.text = "Spheres reached: 0  |  Steps: 0"
            self._obs_label.text = "Obs: [-, -, -, -]  Act: [-, -]"
            self._status_label.text = "Status: Reset complete - Press 'Start'"
        except Exception as e:
            self._status_label.text = f"Status: Reset error - {e}"

    # ------------------------------------------------------------------
    # Physics step callback (runs at 120Hz, policy at 60Hz via decimation)
    # ------------------------------------------------------------------

    def _on_physics_step(self, step_size):
        if not self._is_running or self._jetbot is None or self._policy is None:
            return

        # Decimation: compute policy every 2 physics steps (matching training)
        self._decimation_counter += 1
        if self._decimation_counter % 2 != 0:
            return

        try:
            # Read robot state
            robot_pos, robot_quat = self._jetbot.get_world_pose()  # quat is wxyz
            robot_lin_vel_w = self._jetbot.get_linear_velocity()

            # Compute 4D observation
            obs_np = compute_observations(
                robot_pos, robot_quat, robot_lin_vel_w, self._sphere_pos, self._max_distance
            )
            obs_tensor = torch.from_numpy(obs_np).unsqueeze(0)

            # Normalize (RunningStandardScaler from training)
            obs_normalized = (obs_tensor - self._obs_mean) / self._obs_std

            # Run policy
            with torch.no_grad():
                action = self._policy(obs_normalized).squeeze(0).numpy()

            # Apply wheel velocity targets
            vel_targets = np.zeros(self._jetbot.num_dof)
            vel_targets[self._left_wheel_idx] = float(action[0])
            vel_targets[self._right_wheel_idx] = float(action[1])
            self._jetbot._articulation_view.set_joint_velocity_targets(vel_targets.reshape(1, -1))

            # Check if sphere is reached
            dist_to_sphere = np.linalg.norm(self._sphere_pos[:2] - robot_pos[:2])
            if dist_to_sphere < self._sphere_reach_threshold:
                self._spheres_reached += 1
                self._sphere_pos = random_sphere_position(
                    robot_pos, self._sphere_spawn_min, self._sphere_spawn_max, self._sphere_radius
                )
                set_sphere_position(self._sphere_prim, self._sphere_pos)
                print(
                    f"[SphereFollow] Reached #{self._spheres_reached} - "
                    f"New sphere at [{self._sphere_pos[0]:.2f}, {self._sphere_pos[1]:.2f}]"
                )

            self._step_count += 1

            # Update UI periodically (every ~1 second at 60Hz)
            if self._step_count % 60 == 0:
                self._stats_label.text = (
                    f"Spheres reached: {self._spheres_reached}  |  Steps: {self._step_count}"
                )
                self._obs_label.text = (
                    f"Obs: [{obs_np[0]:+.2f}, {obs_np[1]:+.2f}, {obs_np[2]:.2f}, {obs_np[3]:+.2f}]  "
                    f"Act: [{action[0]:+.3f}, {action[1]:+.3f}]"
                )

        except Exception as e:
            print(f"[SphereFollow] Step error: {e}")
            self._is_running = False
            self._status_label.text = f"Status: Step error - {e}"

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def on_shutdown(self):
        print("[SphereFollow] Extension shutdown")
        self._is_running = False
        if self._world is not None:
            self._world.stop()
        self._window = None

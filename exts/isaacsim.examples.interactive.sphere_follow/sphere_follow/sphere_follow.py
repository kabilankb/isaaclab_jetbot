# SPDX-FileCopyrightText: Copyright (c) 2018-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
JetBot Sphere-Following Policy Deployment Example.

Loads a PPO checkpoint trained in Isaac Lab and runs inference in Isaac Sim.
The JetBot chases a green sphere that repositions when reached.
"""

import math
import os

import numpy as np
import omni
import torch
import torch.nn as nn
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade


# ==============================================================================
# Quaternion utilities (wxyz convention, matching Isaac Lab math_utils)
# ==============================================================================

def quat_rotate(quat_wxyz, vec):
    """Rotate vector by quaternion (wxyz)."""
    xyz = quat_wxyz[1:4]
    t = 2.0 * np.cross(xyz, vec)
    return vec + quat_wxyz[0] * t + np.cross(xyz, t)


def quat_rotate_inverse(quat_wxyz, vec):
    """Rotate vector by inverse quaternion. World -> body frame."""
    xyz = quat_wxyz[1:4]
    t = 2.0 * np.cross(xyz, vec)
    return vec - quat_wxyz[0] * t + np.cross(xyz, t)


# ==============================================================================
# Policy network (matches skrl training architecture exactly)
# ==============================================================================

class PolicyNetwork(nn.Module):
    """Policy: 4 obs -> [Linear(4,64), ELU, Linear(64,64), ELU] -> Linear(64,2)."""

    def __init__(self):
        super().__init__()
        self.net_container = nn.Sequential(
            nn.Linear(4, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
        )
        self.policy_layer = nn.Linear(64, 2)

    def forward(self, obs):
        return self.policy_layer(self.net_container(obs))


# ==============================================================================
# Observation computation (matches SphereFollowEnv._get_observations)
# ==============================================================================

FORWARD_VEC_B = np.array([1.0, 0.0, 0.0])


def compute_observations(robot_pos, robot_quat_wxyz, robot_lin_vel_w, sphere_pos, max_distance=3.0):
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

def create_green_sphere(stage, position, radius=0.1):
    """Create a green sphere visual prim as the target."""
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


def set_sphere_position(sphere_prim, position):
    """Update sphere translate op to new position."""
    for op in sphere_prim.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3f(*position.tolist()))
            return


def random_sphere_position(robot_pos, spawn_min=0.5, spawn_max=1.5, z=0.1):
    """Random position at [spawn_min, spawn_max] distance from robot."""
    angle = np.random.uniform(0, 2 * math.pi)
    dist = np.random.uniform(spawn_min, spawn_max)
    return np.array([
        robot_pos[0] + dist * math.cos(angle),
        robot_pos[1] + dist * math.sin(angle),
        z,
    ])


# ==============================================================================
# SphereFollow BaseSample
# ==============================================================================

class SphereFollow(BaseSample):
    """JetBot sphere-following policy deployment sample.

    Follows the BaseSample lifecycle:
      setup_scene()     -> spawn JetBot, ground, light, green sphere
      setup_post_load() -> load policy checkpoint, register physics callback, play
      on_physics_step() -> read state, compute obs, run policy, apply action, check reach
      setup_pre_reset() -> remove physics callback
      setup_post_reset() -> re-register callback, reposition sphere, play
      world_cleanup()   -> remove callback
    """

    # Default checkpoint path (can be overridden via set_checkpoint_path)
    DEFAULT_CHECKPOINT = os.path.join(
        os.path.expanduser("~"),
        "IsaacLabTutorial",
        "logs", "skrl", "sphere_follow_direct",
        "2026-02-21_19-46-44_ppo_torch", "checkpoints", "best_agent.pt",
    )

    def __init__(self) -> None:
        super().__init__()
        # Match training: physics at 120Hz, render at 60Hz (decimation=2)
        self._world_settings["physics_dt"] = 1.0 / 120.0
        self._world_settings["rendering_dt"] = 1.0 / 60.0
        self._world_settings["stage_units_in_meters"] = 1.0

        # Policy state
        self._policy = None
        self._obs_mean = None
        self._obs_std = None
        self._checkpoint_path = self.DEFAULT_CHECKPOINT

        # Scene objects
        self._jetbot = None
        self._sphere_prim = None

        # Runtime state
        self._sphere_pos = np.array([0.8, 0.3, 0.1])
        self._spheres_reached = 0
        self._step_count = 0
        self._physics_ready = False
        self._decimation_counter = 0

        # Config (matching SphereFollowEnvCfg)
        self._sphere_reach_threshold = 0.3
        self._sphere_spawn_min = 0.5
        self._sphere_spawn_max = 1.5
        self._sphere_radius = 0.1
        self._max_distance = 3.0

        # Wheel indices (set after load)
        self._left_wheel_idx = 0
        self._right_wheel_idx = 1

        # Callbacks for UI status updates
        self._on_status_update = None
        self._last_obs = None
        self._last_action = None

    def set_checkpoint_path(self, path):
        """Set checkpoint path before loading world."""
        self._checkpoint_path = path

    def get_spheres_reached(self):
        return self._spheres_reached

    def get_step_count(self):
        return self._step_count

    def get_last_obs(self):
        return self._last_obs

    def get_last_action(self):
        return self._last_action

    def set_status_callback(self, callback):
        """Set a callback(spheres_reached, step_count, obs, action) for UI updates."""
        self._on_status_update = callback

    # ------------------------------------------------------------------
    # Policy loading
    # ------------------------------------------------------------------

    def _load_policy(self):
        """Load policy weights and observation normalizer from skrl checkpoint."""
        if not os.path.exists(self._checkpoint_path):
            print(f"[SphereFollow] ERROR: Checkpoint not found: {self._checkpoint_path}")
            return False

        ckpt = torch.load(self._checkpoint_path, map_location="cpu", weights_only=False)

        self._policy = PolicyNetwork()
        state_dict = {
            key: ckpt["policy"][key]
            for key in [
                "net_container.0.weight", "net_container.0.bias",
                "net_container.2.weight", "net_container.2.bias",
                "policy_layer.weight", "policy_layer.bias",
            ]
        }
        self._policy.load_state_dict(state_dict)
        self._policy.eval()

        self._obs_mean = ckpt["state_preprocessor"]["running_mean"].float()
        obs_var = ckpt["state_preprocessor"]["running_variance"].float()
        self._obs_std = torch.sqrt(obs_var + 1e-8)

        print(f"[SphereFollow] Policy loaded from {self._checkpoint_path}")
        return True

    # ------------------------------------------------------------------
    # BaseSample lifecycle
    # ------------------------------------------------------------------

    def setup_scene(self):
        world = self.get_world()

        # Ground plane
        world.scene.add_default_ground_plane()

        # JetBot
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            print("[SphereFollow] ERROR: Could not find Isaac Sim assets folder")
            return
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"

        self._jetbot = world.scene.add(
            WheeledRobot(
                prim_path="/World/Jetbot",
                name="sphere_follow_jetbot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_asset_path,
                position=np.array([0, 0.0, 0.05]),
            )
        )

        # Dome light (matching training environment)
        stage = omni.usd.get_context().get_stage()
        light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        light.CreateIntensityAttr().Set(2000.0)
        light.CreateColorAttr().Set(Gf.Vec3f(0.75, 0.75, 0.75))

        # Green target sphere
        self._sphere_pos = np.array([0.8, 0.3, 0.1])
        self._sphere_prim = create_green_sphere(stage, self._sphere_pos, radius=self._sphere_radius)

        # Camera for good view
        set_camera_view(
            eye=[2.0, 2.0, 1.5],
            target=[0.0, 0.0, 0.0],
            camera_prim_path="/OmniverseKit_Persp",
        )

    async def setup_post_load(self):
        self._physics_ready = False
        self._spheres_reached = 0
        self._step_count = 0
        self._decimation_counter = 0

        # Load the trained policy
        if not self._load_policy():
            print("[SphereFollow] WARNING: Policy not loaded. Robot will not move.")

        # Get wheel joint indices
        self._left_wheel_idx = self._jetbot.dof_names.index("left_wheel_joint")
        self._right_wheel_idx = self._jetbot.dof_names.index("right_wheel_joint")

        # Register physics callback
        if not self.get_world().physics_callback_exists("sphere_follow_step"):
            self.get_world().add_physics_callback("sphere_follow_step", callback_fn=self.on_physics_step)

        # Start playing immediately (like franka example)
        await self.get_world().play_async()

    async def setup_pre_reset(self):
        # Remove physics callback before reset
        world = self.get_world()
        if world.physics_callback_exists("sphere_follow_step"):
            world.remove_physics_callback("sphere_follow_step")

    async def setup_post_reset(self):
        self._physics_ready = False
        self._spheres_reached = 0
        self._step_count = 0
        self._decimation_counter = 0

        # Reposition sphere
        robot_pos, _ = self._jetbot.get_world_pose()
        self._sphere_pos = random_sphere_position(
            robot_pos, self._sphere_spawn_min, self._sphere_spawn_max, self._sphere_radius
        )
        set_sphere_position(self._sphere_prim, self._sphere_pos)

        # Re-register physics callback
        if not self.get_world().physics_callback_exists("sphere_follow_step"):
            self.get_world().add_physics_callback("sphere_follow_step", callback_fn=self.on_physics_step)

        await self.get_world().play_async()

    async def setup_post_clear(self):
        self._jetbot = None
        self._sphere_prim = None
        self._policy = None

    def world_cleanup(self):
        world = self.get_world()
        if world is not None and world.physics_callback_exists("sphere_follow_step"):
            world.remove_physics_callback("sphere_follow_step")

    # ------------------------------------------------------------------
    # Physics step (runs at 120Hz, policy applied at 60Hz via decimation)
    # ------------------------------------------------------------------

    def on_physics_step(self, step_size):
        if self._jetbot is None:
            return

        # First step: mark as ready
        if not self._physics_ready:
            self._physics_ready = True
            return

        # No policy loaded - do nothing
        if self._policy is None:
            return

        # Decimation: apply policy every 2 physics steps (matching training)
        self._decimation_counter += 1
        if self._decimation_counter % 2 != 0:
            return

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

        # Run policy forward pass
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
        self._last_obs = obs_np
        self._last_action = action

        # Notify UI callback periodically
        if self._on_status_update and self._step_count % 30 == 0:
            self._on_status_update(self._spheres_reached, self._step_count, obs_np, action)

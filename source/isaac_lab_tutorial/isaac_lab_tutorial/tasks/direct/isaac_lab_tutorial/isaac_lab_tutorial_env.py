# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from .isaac_lab_tutorial_env_cfg import IsaacLabTutorialEnvCfg, SphereFollowEnvCfg

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)

class IsaacLabTutorialEnv(DirectRLEnv):
    cfg: IsaacLabTutorialEnvCfg

    def __init__(self, cfg: IsaacLabTutorialEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.visualization_markers = define_markers()

        self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()  
        self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda()
        self.commands = torch.randn((self.cfg.scene.num_envs, 3)).cuda()
        self.commands[:,-1] = 0.0
        self.commands = self.commands/torch.linalg.norm(self.commands, dim=1, keepdim=True)
        
        # offsets to account for atan range and keep things on [-pi, pi]
        ratio = self.commands[:,1]/(self.commands[:,0]+1E-8)
        gzero = torch.where(self.commands > 0, True, False)
        lzero = torch.where(self.commands < 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

        self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset[:,-1] = 0.5
        self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
        self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
        

    def _visualize_markers(self):
        self.marker_locations = self.robot.data.root_pos_w
        self.forward_marker_orientations = self.robot.data.root_quat_w
        self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()

        loc = self.marker_locations + self.marker_offset
        loc = torch.vstack((loc, loc))
        rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations))

        all_envs = torch.arange(self.cfg.scene.num_envs)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))

        self.visualization_markers.visualize(loc, rots, marker_indices=indices)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()# + torch.ones_like(actions)
        self._visualize_markers()

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        self.velocity = self.robot.data.root_com_vel_w 
        self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)
        # obs = torch.hstack((self.velocity, self.commands))

        dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        cross = torch.cross(self.forwards, self.commands, dim=-1)[:,-1].reshape(-1,1)
        forward_speed = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        obs = torch.hstack((dot, cross, forward_speed))
        
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        forward_reward = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        alignment_reward = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        total_reward = forward_reward + alignment_reward
        # total_reward = forward_reward*alignment_reward
        # total_reward = forward_reward*alignment_reward + forward_reward
        # total_reward = forward_reward*torch.exp(alignment_reward)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return False, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self.commands[env_ids] = torch.randn((len(env_ids), 3)).cuda()
        self.commands[env_ids,-1] = 0.0
        self.commands[env_ids] = self.commands[env_ids]/torch.linalg.norm(self.commands[env_ids], dim=1, keepdim=True)
        
        ratio = self.commands[env_ids][:,1]/(self.commands[env_ids][:,0]+1E-8)
        gzero = torch.where(self.commands[env_ids] > 0, True, False)
        lzero = torch.where(self.commands[env_ids]< 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws[env_ids] = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_state_to_sim(default_root_state, env_ids)
        self._visualize_markers()


def define_sphere_marker(radius: float = 0.1) -> VisualizationMarkers:
    """Define a green sphere marker for the target."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/sphereMarkers",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


class SphereFollowEnv(DirectRLEnv):
    cfg: SphereFollowEnvCfg

    def __init__(self, cfg: SphereFollowEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # sphere marker
        self.sphere_markers = define_sphere_marker(self.cfg.sphere_radius)

        # sphere positions in world frame (one per env)
        self.sphere_pos_w = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)
        self.prev_dist_to_sphere = torch.zeros((self.cfg.scene.num_envs,), device=self.device)

    def _spawn_sphere_positions(self, env_ids: torch.Tensor, robot_pos_xy: torch.Tensor | None = None) -> None:
        """Spawn spheres at random positions relative to the robot.

        Args:
            env_ids: Environment indices to respawn spheres for.
            robot_pos_xy: Optional (N, 2) tensor of robot XY positions. If None,
                reads from sim data (only valid mid-episode, NOT right after reset).
        """
        num = len(env_ids)
        # random angle and distance
        angles = torch.rand(num, device=self.device) * 2.0 * math.pi
        dists = (
            torch.rand(num, device=self.device)
            * (self.cfg.sphere_spawn_range_max - self.cfg.sphere_spawn_range_min)
            + self.cfg.sphere_spawn_range_min
        )
        # use provided position or read from sim
        if robot_pos_xy is None:
            robot_pos_xy = self.robot.data.root_pos_w[env_ids, :2]
        offset_x = dists * torch.cos(angles)
        offset_y = dists * torch.sin(angles)
        self.sphere_pos_w[env_ids, 0] = robot_pos_xy[:, 0] + offset_x
        self.sphere_pos_w[env_ids, 1] = robot_pos_xy[:, 1] + offset_y
        self.sphere_pos_w[env_ids, 2] = self.cfg.sphere_radius

    def _visualize_sphere(self):
        """Update sphere marker positions."""
        orientations = torch.zeros((self.cfg.scene.num_envs, 4), device=self.device)
        orientations[:, 0] = 1.0  # identity quaternion (w, x, y, z)
        self.sphere_markers.visualize(self.sphere_pos_w, orientations)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self._visualize_sphere()

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        # forward direction of robot
        forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)
        # direction to sphere (XY plane)
        dir_to_sphere = self.sphere_pos_w - self.robot.data.root_pos_w
        dir_to_sphere[:, 2] = 0.0
        dist = torch.linalg.norm(dir_to_sphere[:, :2], dim=1, keepdim=True).clamp(min=1e-6)
        dir_to_sphere_norm = dir_to_sphere / dist

        # dot product: +1 facing sphere, -1 facing away
        dot = torch.sum(forwards * dir_to_sphere_norm, dim=-1, keepdim=True)
        # cross product Z component: sign indicates turn direction
        cross_z = (forwards[:, 0] * dir_to_sphere_norm[:, 1] - forwards[:, 1] * dir_to_sphere_norm[:, 0]).unsqueeze(-1)
        # normalized distance
        dist_norm = (dist / self.cfg.sphere_max_distance).clamp(0.0, 1.0)
        # forward speed in body frame
        forward_speed = self.robot.data.root_com_lin_vel_b[:, 0].unsqueeze(-1)

        obs = torch.cat((dot, cross_z, dist_norm, forward_speed), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # current distance to sphere
        diff = self.sphere_pos_w[:, :2] - self.robot.data.root_pos_w[:, :2]
        curr_dist = torch.linalg.norm(diff, dim=1)

        # approach reward: positive when getting closer
        approach = (self.prev_dist_to_sphere - curr_dist) * self.cfg.approach_reward_scale

        # alignment reward
        forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)
        dir_to_sphere = self.sphere_pos_w - self.robot.data.root_pos_w
        dir_to_sphere[:, 2] = 0.0
        dist_for_norm = torch.linalg.norm(dir_to_sphere[:, :2], dim=1, keepdim=True).clamp(min=1e-6)
        dir_to_sphere_norm = dir_to_sphere / dist_for_norm
        dot = torch.sum(forwards * dir_to_sphere_norm, dim=-1)
        alignment = dot * self.cfg.alignment_reward_scale

        # reach bonus: sphere reached, reposition it
        reached = curr_dist < self.cfg.sphere_reach_threshold
        reach = reached.float() * self.cfg.reach_bonus

        # reposition sphere for envs that reached it
        reached_ids = torch.where(reached)[0]
        if len(reached_ids) > 0:
            self._spawn_sphere_positions(reached_ids)
            # reset prev_dist for repositioned spheres to avoid reward spike
            new_diff = self.sphere_pos_w[reached_ids, :2] - self.robot.data.root_pos_w[reached_ids, :2]
            curr_dist[reached_ids] = torch.linalg.norm(new_diff, dim=1)

        # time penalty
        time_pen = torch.full_like(curr_dist, self.cfg.time_penalty)

        # update prev_dist
        self.prev_dist_to_sphere = curr_dist.clone()

        total_reward = approach + alignment + reach + time_pen
        return total_reward.unsqueeze(-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return False, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # reset robot pose
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(default_root_state, env_ids)

        # use the known reset position (sim data is stale until next step)
        env_ids_tensor = torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
        robot_reset_pos_xy = default_root_state[:, :2]

        # spawn new sphere positions relative to the reset position
        self._spawn_sphere_positions(env_ids_tensor, robot_pos_xy=robot_reset_pos_xy)

        # init prev_dist using the known reset position
        diff = self.sphere_pos_w[env_ids, :2] - robot_reset_pos_xy
        self.prev_dist_to_sphere[env_ids] = torch.linalg.norm(diff, dim=1)

        self._visualize_sphere()

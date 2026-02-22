"""Generate a Medium-style blog post PDF about JetBot Sphere-Following with Isaac Lab, Isaac Sim Standalone, and Isaac Sim Extension."""

from fpdf import FPDF


class MediumBlogPDF(FPDF):
    """Custom PDF styled like a Medium blog post."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def add_title(self, title):
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(41, 41, 41)
        self.multi_cell(0, 14, title)
        self.ln(4)

    def add_subtitle(self, subtitle):
        self.set_font("Helvetica", "", 16)
        self.set_text_color(117, 117, 117)
        self.multi_cell(0, 9, subtitle)
        self.ln(4)

    def add_author_line(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(117, 117, 117)
        self.cell(0, 8, text)
        self.ln(12)

    def add_divider(self):
        self.ln(4)
        x = self.get_x()
        y = self.get_y()
        self.set_draw_color(200, 200, 200)
        self.line(x + 60, y, x + 130, y)
        self.ln(8)

    def add_chapter_title(self, text):
        """Large chapter heading with a colored bar."""
        self.add_page()
        self.ln(20)
        # Colored bar
        x = self.get_x()
        y = self.get_y()
        self.set_fill_color(0, 120, 212)
        self.rect(x, y, self.w - 2 * self.l_margin, 3, style="F")
        self.ln(10)
        self.set_font("Helvetica", "B", 26)
        self.set_text_color(41, 41, 41)
        self.multi_cell(0, 13, text)
        self.ln(6)

    def add_heading(self, text):
        self.ln(6)
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(41, 41, 41)
        self.multi_cell(0, 11, text)
        self.ln(3)

    def add_subheading(self, text):
        self.ln(3)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(41, 41, 41)
        self.multi_cell(0, 9, text)
        self.ln(2)

    def add_body(self, text):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(41, 41, 41)
        self.multi_cell(0, 7, text)
        self.ln(3)

    def add_bold_body(self, text):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(41, 41, 41)
        self.multi_cell(0, 7, text)
        self.ln(3)

    def add_code_block(self, code):
        self.ln(2)
        self.set_fill_color(246, 246, 246)
        self.set_font("Courier", "", 9)
        self.set_text_color(41, 41, 41)
        lines = code.strip().split("\n")
        # Calculate block height
        block_h = len(lines) * 5.5 + 8
        x = self.get_x()
        y = self.get_y()
        # Check page break
        if y + block_h > self.h - self.b_margin:
            self.add_page()
            y = self.get_y()
        self.rect(x, y, self.w - 2 * self.l_margin, block_h, style="F")
        self.ln(4)
        for line in lines:
            self.cell(4)
            self.cell(0, 5.5, line)
            self.ln()
        self.ln(6)

    def add_bullet(self, text):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(41, 41, 41)
        x = self.get_x()
        self.cell(6)
        self.cell(4, 7, "-")
        self.multi_cell(self.w - 2 * self.l_margin - 10, 7, text)
        self.ln(1)

    def add_table(self, headers, rows):
        self.ln(2)
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(41, 41, 41)
        col_w = (self.w - 2 * self.l_margin) / len(headers)
        for h in headers:
            self.cell(col_w, 8, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 9)
        for row in rows:
            for cell in row:
                self.cell(col_w, 7, str(cell), border=1, align="C")
            self.ln()
        self.ln(4)

    def add_quote(self, text):
        self.ln(2)
        x = self.get_x()
        y = self.get_y()
        self.set_draw_color(41, 41, 41)
        self.set_line_width(0.8)
        self.line(x, y, x, y + 14)
        self.set_line_width(0.2)
        self.cell(6)
        self.set_font("Helvetica", "I", 12)
        self.set_text_color(80, 80, 80)
        self.multi_cell(self.w - 2 * self.l_margin - 6, 7, text)
        self.ln(4)


def build_pdf():
    pdf = MediumBlogPDF()
    pdf.alias_nb_pages()
    pdf.set_left_margin(25)
    pdf.set_right_margin(25)

    # ====================================================================
    # COVER PAGE
    # ====================================================================
    pdf.add_page()
    pdf.ln(15)
    pdf.add_title("Training a JetBot on Isaac Lab with the Dell Pro Max Powered by NVIDIA RTX PRO 6000 Blackwell")
    pdf.add_subtitle(
        "Building a reinforcement learning environment from scratch where a "
        "differential-drive robot learns to chase a moving sphere using PPO - "
        "trained in minutes on Blackwell-class GPU hardware. From Isaac Lab training "
        "to Isaac Sim standalone deployment to a full Isaac Sim extension."
    )
    pdf.add_author_line("February 2026  |  18 min read")
    pdf.add_divider()

    # ---- Introduction ----
    pdf.add_body(
        "Imagine dropping a small wheeled robot into an arena with a glowing green sphere. "
        "The robot knows nothing about the world - no maps, no hard-coded rules, no PID "
        "controllers. All it has is a neural network brain and a reward signal that says: "
        '"get closer to the sphere." Within minutes of simulated training across 100 parallel '
        "worlds, the robot figures out how to spin its wheels to chase the target, and when it "
        "catches up, the sphere teleports to a new random location and the hunt begins again."
    )
    pdf.add_body(
        "This is the power of reinforcement learning (RL) in simulation. Running on a "
        "Dell Pro Max workstation equipped with the NVIDIA RTX PRO 6000 Blackwell GPU "
        "(96 GB VRAM), we can spin up 100 parallel physics environments and train a "
        "policy in roughly 10 minutes."
    )
    pdf.add_body(
        "This document covers the complete journey in three chapters:"
    )
    pdf.add_bullet(
        "Chapter I: Isaac Lab - Designing the RL environment, observations, rewards, "
        "and training the PPO policy across 100 parallel environments."
    )
    pdf.add_bullet(
        "Chapter II: Isaac Sim Standalone - Deploying the trained checkpoint in a "
        "standalone Isaac Sim script outside the Isaac Lab framework."
    )
    pdf.add_bullet(
        "Chapter III: Isaac Sim Extension - Packaging the deployment as a proper "
        "Isaac Sim extension with UI controls, registered in the Examples Browser."
    )

    # ====================================================================
    # CHAPTER I: ISAAC LAB
    # ====================================================================
    pdf.add_chapter_title("Chapter I: Isaac Lab - Training the Policy")

    pdf.add_body(
        "In this chapter we build the complete training pipeline: defining the robot, "
        "the sphere-following task, ego-centric observations, a shaped reward function, "
        "and training with PPO using the skrl library."
    )

    # ---- 1.1 ----
    pdf.add_heading("1.1 The Platform: Isaac Lab + JetBot")

    pdf.add_subheading("What is Isaac Lab?")
    pdf.add_body(
        "Isaac Lab is NVIDIA's open-source framework for robot learning built on top of "
        "Isaac Sim and the Omniverse platform. It provides GPU-accelerated physics simulation, "
        "parallel environment execution, and first-class integration with RL libraries like "
        "skrl, rl_games, RSL-RL, and Stable Baselines3."
    )
    pdf.add_body(
        "The key advantage is scale: Isaac Lab can run hundreds or thousands of environments "
        "in parallel on a single GPU. Each environment is a full physics simulation with "
        "articulated robots, collision geometry, and contact forces - all running at 120 Hz."
    )

    pdf.add_subheading("The JetBot: A Differential-Drive Robot")
    pdf.add_body(
        "The NVIDIA JetBot is a two-wheeled differential-drive robot. It has exactly two "
        "actuated joints - a left wheel and a right wheel. By varying the velocity of each "
        "wheel independently, the robot can move forward, turn left, turn right, spin in "
        "place, or move in curved arcs."
    )
    pdf.add_body("This simplicity makes it an ideal testbed for RL:")
    pdf.add_bullet("Action space: 2 continuous values (left wheel velocity, right wheel velocity)")
    pdf.add_bullet("Dynamics: nonholonomic - the robot cannot move sideways directly")
    pdf.add_bullet("Challenge: the agent must learn coordinated wheel commands to steer toward a goal")

    pdf.add_subheading("Environment Configuration")
    pdf.add_body("The environment configuration defines the simulation parameters:")
    pdf.add_code_block(
        """@configclass
class SphereFollowEnvCfg(DirectRLEnvCfg):
    decimation = 2                    # control at 60 Hz
    episode_length_s = 20.0           # 20-second episodes
    action_space = 2                  # left/right wheel velocity
    observation_space = 4             # see Section 1.3
    env_spacing = 4.0                 # meters between envs
    sphere_radius = 0.1              # 10cm green sphere
    sphere_reach_threshold = 0.3     # "reached" at 30cm
    sphere_spawn_range_min = 0.5     # spawn 0.5-1.5m away
    sphere_spawn_range_max = 1.5"""
    )

    # ---- 1.2 ----
    pdf.add_heading("1.2 The Task: Sphere Following")
    pdf.add_body(
        "The task is conceptually simple: a green sphere spawns at a random position near "
        "the robot. The robot must navigate to the sphere. When it gets within 30cm, the "
        "sphere teleports to a new random location and the robot must chase it again. This "
        "cycle continues for the entire 20-second episode."
    )
    pdf.add_body(
        "The sphere is implemented as a visualization marker - it has no physics collider "
        "and no mass. Its position is controlled entirely by the environment code."
    )
    pdf.add_body("Each episode proceeds as follows:")
    pdf.add_bullet("Robot resets to its default pose at the center of the env cell")
    pdf.add_bullet("Sphere spawns at a random angle, 0.5-1.5m from the robot")
    pdf.add_bullet("Robot takes actions (wheel velocities) based on observations")
    pdf.add_bullet("When robot reaches sphere (dist < 0.3m): +5.0 bonus, sphere respawns")
    pdf.add_bullet("Episode ends after 20 seconds (timeout)")

    # ---- 1.3 ----
    pdf.add_heading("1.3 Observation Space Design")
    pdf.add_body(
        "Designing good observations is one of the most critical decisions in RL environment "
        "design. Our observation is a 4-dimensional vector computed every step:"
    )
    pdf.add_code_block("""obs = [dot, cross_z, dist_norm, forward_speed]""")

    pdf.add_bold_body("1. Dot Product (alignment signal)")
    pdf.add_body(
        "dot = sum(forward_dir * dir_to_sphere). Equals +1 when the robot faces "
        "directly toward the sphere, -1 when facing away. It tells the agent HOW "
        "WELL it is aligned with the target."
    )
    pdf.add_bold_body("2. Cross Product Z-component (steering signal)")
    pdf.add_body(
        "cross_z = (forward x dir_to_sphere).z. Positive when the sphere is to the "
        "left, negative when to the right. Combined with the dot product, these two "
        "signals fully encode the relative bearing to the sphere."
    )
    pdf.add_bold_body("3. Normalized Distance")
    pdf.add_body(
        "dist_norm = distance / 3.0, clamped to [0, 1]. Tells the agent HOW FAR "
        "the sphere is, normalized to a neural-network-friendly range."
    )
    pdf.add_bold_body("4. Forward Speed")
    pdf.add_body(
        "forward_speed = body-frame X velocity. Gives the agent proprioceptive "
        "feedback about its current motion."
    )
    pdf.add_quote(
        "These four numbers are sufficient for the agent to solve the task. "
        "The ego-centric representation generalizes better across environments "
        "than raw positions or quaternions."
    )

    # ---- 1.4 ----
    pdf.add_heading("1.4 Reward Function Engineering")
    pdf.add_body("Our reward is a sum of four components:")
    pdf.add_code_block("""reward = approach + alignment + reach_bonus + time_penalty""")

    pdf.add_subheading("Approach Reward (dense, distance-based)")
    pdf.add_code_block("""approach = (prev_dist - curr_dist) * 1.0""")
    pdf.add_body(
        "A potential-based shaping reward. Every step, we compare the current distance "
        "to the previous distance. If the robot moved closer, the reward is positive."
    )
    pdf.add_subheading("Alignment Reward (orientation incentive)")
    pdf.add_code_block("""alignment = dot(forward, dir_to_sphere) * 0.5""")
    pdf.add_body(
        "Rewards the robot for facing the sphere, even before it starts moving. "
        "Encourages turn-then-drive behavior."
    )
    pdf.add_subheading("Reach Bonus (sparse, goal-based)")
    pdf.add_code_block("""reach = 5.0 if dist < 0.3m else 0.0""")
    pdf.add_body(
        "A one-time bonus when the robot reaches the sphere. After collecting it, "
        "the sphere respawns at a new random location."
    )
    pdf.add_subheading("Time Penalty (efficiency pressure)")
    pdf.add_code_block("""time_penalty = -0.01 per step""")
    pdf.add_body(
        "A small constant penalty discouraging the agent from spinning or sitting still."
    )

    # ---- 1.5 ----
    pdf.add_heading("1.5 The Control System")
    pdf.add_body(
        "The trained policy is a neural network that maps observations to actions. "
        "At each control step (60 Hz), the following pipeline executes:"
    )
    pdf.add_code_block(
        """1. Physics sim advances (120 Hz, decimation=2)
2. Read robot state: position, orientation, velocity
3. Compute 4D observation vector
4. Neural network forward pass: obs -> actions
5. Actions = [left_wheel_vel, right_wheel_vel]
6. Apply velocity targets to wheel joints
7. Compute reward, check termination"""
    )

    pdf.add_subheading("Network Architecture")
    pdf.add_code_block(
        """Input:  4 observations
  |
  v
Linear(4 -> 64) + ELU
Linear(64 -> 64) + ELU
  |            |
  v            v
Policy:      Value:
Linear(64->2) Linear(64->1)"""
    )

    # ---- 1.6 ----
    pdf.add_heading("1.6 Training with PPO")

    pdf.add_table(
        ["Parameter", "Value", "Purpose"],
        [
            ["Environments", "100", "Parallel data collection"],
            ["Network", "[64, 64]", "Shared policy-value backbone"],
            ["Rollout length", "48 steps", "Experience per update"],
            ["Learning rate", "3e-4", "Adaptive via KL threshold"],
            ["Mini-batches", "8", "SGD batches per epoch"],
            ["Learning epochs", "8", "Passes over rollout data"],
            ["Entropy coeff", "0.01", "Exploration bonus"],
            ["Total timesteps", "24,000", "~10 min on RTX PRO 6000"],
            ["Discount factor", "0.99", "Long-horizon returns"],
        ],
    )

    pdf.add_subheading("Training Progression")
    pdf.add_body(
        "Phase 1 - Random exploration: The agent outputs near-random wheel velocities. "
        "The approach reward provides gradient signal even during random motion."
    )
    pdf.add_body(
        "Phase 2 - Turning behavior emerges: The agent learns that facing the sphere "
        "yields positive alignment reward."
    )
    pdf.add_body(
        "Phase 3 - Drive-and-chase: The agent combines turning with forward motion, "
        "reaching spheres more frequently."
    )
    pdf.add_body(
        "Phase 4 - Refined tracking: The agent learns smooth, efficient curved "
        "trajectories directly to the target."
    )

    # ---- 1.7 ----
    pdf.add_heading("1.7 Checkpoint and Key Implementation Details")

    pdf.add_body("During training, skrl saves checkpoints and tracks the best agent:")
    pdf.add_code_block(
        """logs/skrl/sphere_follow_direct/
  2026-02-21_19-46-44_ppo_torch/
    checkpoints/
      best_agent.pt      # best by episode return
    params/
      env.yaml           # environment config
      agent.yaml         # agent hyperparameters"""
    )

    pdf.add_subheading("What the Checkpoint Contains")
    pdf.add_bullet("Policy network weights and biases (4 -> 64 -> 64 -> 2)")
    pdf.add_bullet("Value network weights (shared backbone + value head)")
    pdf.add_bullet("Running mean/std for observation normalization (RunningStandardScaler)")
    pdf.add_bullet("Optimizer state (for resuming training)")

    pdf.add_subheading("Sphere Repositioning Without Reward Spikes")
    pdf.add_body(
        "When the robot reaches a sphere, we reposition it. A naive implementation "
        "would cause a reward spike. Our solution: reset prev_dist immediately:"
    )
    pdf.add_code_block(
        """reached_ids = torch.where(curr_dist < 0.3)[0]
if len(reached_ids) > 0:
    self._spawn_sphere_positions(reached_ids)
    new_diff = sphere_pos[reached_ids] - robot_pos[reached_ids]
    curr_dist[reached_ids] = torch.linalg.norm(new_diff)
self.prev_dist_to_sphere = curr_dist.clone()"""
    )

    pdf.add_subheading("Stale Simulation Data During Reset")
    pdf.add_body(
        "After write_root_state_to_sim(), robot position data is not updated until the "
        "next simulation step. We pass the known reset position directly:"
    )
    pdf.add_code_block(
        """robot_reset_pos_xy = default_root_state[:, :2]
self._spawn_sphere_positions(env_ids,
    robot_pos_xy=robot_reset_pos_xy)"""
    )

    # ---- 1.8 ----
    pdf.add_heading("1.8 Isaac Lab Launch Commands")

    pdf.add_code_block(
        """# Install the package
./launch.sh install

# Verify environment registration
./launch.sh list-envs

# Visual smoke test with random agent
./launch.sh random-agent --num_envs 10

# Train the PPO agent (100 parallel envs)
./launch.sh train --algorithm PPO --num_envs 100

# Evaluate the trained agent
./launch.sh play --checkpoint \\
    logs/skrl/sphere_follow_direct/<run>/checkpoints/best_agent.pt

# Quick reference
./launch.sh install          # Install package
./launch.sh list-envs        # List registered envs
./launch.sh train            # Train RL agent
./launch.sh play             # Evaluate trained agent
./launch.sh help             # Show all options"""
    )

    # ====================================================================
    # CHAPTER II: ISAAC SIM STANDALONE
    # ====================================================================
    pdf.add_chapter_title("Chapter II: Isaac Sim Standalone - Deploying the Trained Policy")

    pdf.add_body(
        "Once we have a trained checkpoint from Isaac Lab, we can deploy it in a "
        "standalone Isaac Sim script - no Isaac Lab framework required. This is useful "
        "for demos, integration testing, or running the policy on machines that have "
        "Isaac Sim but not the full Isaac Lab stack."
    )

    # ---- 2.1 ----
    pdf.add_heading("2.1 From Isaac Lab to Isaac Sim: The API Mapping")

    pdf.add_body(
        "Isaac Lab provides high-level abstractions (Articulation, DirectRLEnv, math_utils) "
        "that wrap lower-level Isaac Sim APIs. When moving to standalone Isaac Sim, we need "
        "to map each Isaac Lab call to its Isaac Sim equivalent:"
    )

    pdf.add_table(
        ["Isaac Lab API", "Isaac Sim Standalone"],
        [
            ["robot.data.root_pos_w", "jetbot.get_world_pose()[0]"],
            ["robot.data.root_link_quat_w", "jetbot.get_world_pose()[1]"],
            ["robot.data.root_com_lin_vel_b", "quat_rotate_inverse(q, get_linear_velocity())"],
            ["math_utils.quat_apply(q, v)", "quat_rotate(q, v)"],
            ["robot.set_joint_velocity_target", "articulation_view.set_joint_velocity_targets"],
            ["sim dt=1/120, decimation=2", "physics_dt=1/120, rendering_dt=1/60"],
        ],
    )

    # ---- 2.2 ----
    pdf.add_heading("2.2 Standalone Script Architecture")

    pdf.add_body(
        "The standalone script (scripts/standalone_sphere_follow.py) follows this "
        "structure:"
    )
    pdf.add_bullet("Create SimulationApp (must happen before any Isaac Sim imports)")
    pdf.add_bullet("Define quaternion utilities (quat_rotate, quat_rotate_inverse) for wxyz convention")
    pdf.add_bullet("Define PolicyNetwork matching the training architecture exactly")
    pdf.add_bullet("Load checkpoint: extract policy weights + RunningStandardScaler parameters")
    pdf.add_bullet("Create World with matching physics/render rates")
    pdf.add_bullet("Spawn JetBot from USD asset, ground plane, dome light, green sphere")
    pdf.add_bullet("Main loop: read state, compute obs, normalize, run policy, apply action")

    pdf.add_subheading("Quaternion Utilities (wxyz Convention)")
    pdf.add_body(
        "Both Isaac Lab and Isaac Sim use the wxyz (scalar-first) quaternion convention. "
        "We implement quat_rotate and quat_rotate_inverse using the cross-product formula "
        "to match Isaac Lab's math_utils exactly:"
    )
    pdf.add_code_block(
        """def quat_rotate(quat_wxyz, vec):
    \"\"\"Rotate vector by quaternion (wxyz).\"\"\"
    xyz = quat_wxyz[1:4]
    t = 2.0 * np.cross(xyz, vec)
    return vec + quat_wxyz[0] * t + np.cross(xyz, t)

def quat_rotate_inverse(quat_wxyz, vec):
    \"\"\"World frame -> body frame.\"\"\"
    xyz = quat_wxyz[1:4]
    t = 2.0 * np.cross(xyz, vec)
    return vec - quat_wxyz[0] * t + np.cross(xyz, t)"""
    )

    pdf.add_subheading("Policy Network (Exact Architecture Match)")
    pdf.add_body(
        "The PolicyNetwork must exactly match the skrl training architecture. "
        "We extract only the policy-relevant weights from the checkpoint:"
    )
    pdf.add_code_block(
        """class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_container = nn.Sequential(
            nn.Linear(4, 64), nn.ELU(),
            nn.Linear(64, 64), nn.ELU(),
        )
        self.policy_layer = nn.Linear(64, 2)

# Load from skrl checkpoint:
state_dict = {
    key: ckpt["policy"][key]
    for key in ["net_container.0.weight", ...]
}
# Also extract observation normalizer:
obs_mean = ckpt["state_preprocessor"]["running_mean"]
obs_std = sqrt(ckpt["state_preprocessor"]["running_variance"] + 1e-8)"""
    )

    pdf.add_subheading("Observation Normalization")
    pdf.add_body(
        "During training, skrl uses a RunningStandardScaler to normalize observations. "
        "The checkpoint stores the running mean and variance. At inference time, we must "
        "apply the same normalization:"
    )
    pdf.add_code_block(
        """obs_normalized = (obs_tensor - obs_mean) / obs_std"""
    )
    pdf.add_body(
        "Without this normalization step, the policy outputs would be meaningless - "
        "the network was trained on normalized inputs."
    )

    # ---- 2.3 ----
    pdf.add_heading("2.3 The Main Loop")

    pdf.add_body(
        "The standalone main loop mirrors the training environment's step logic:"
    )
    pdf.add_code_block(
        """while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_playing():
        # Read robot state
        robot_pos, robot_quat = jetbot.get_world_pose()
        robot_lin_vel_w = jetbot.get_linear_velocity()

        # Compute 4D observation
        obs = compute_observations(
            robot_pos, robot_quat, robot_lin_vel_w,
            sphere_pos, max_distance=3.0)

        # Normalize + policy forward pass
        obs_norm = (torch.from_numpy(obs) - obs_mean) / obs_std
        action = policy(obs_norm).squeeze().numpy()

        # Apply wheel velocity targets
        vel_targets = np.zeros(jetbot.num_dof)
        vel_targets[left_idx] = action[0]
        vel_targets[right_idx] = action[1]
        jetbot._articulation_view.set_joint_velocity_targets(
            vel_targets.reshape(1, -1))

        # Check sphere reached -> reposition
        if np.linalg.norm(sphere_pos[:2] - robot_pos[:2]) < 0.3:
            sphere_pos = random_sphere_position(robot_pos)
            set_sphere_position(sphere_prim, sphere_pos)"""
    )

    # ---- 2.4 ----
    pdf.add_heading("2.4 Running the Standalone Script")

    pdf.add_code_block(
        """python scripts/standalone_sphere_follow.py \\
    --checkpoint logs/skrl/sphere_follow_direct/\\
    <run>/checkpoints/best_agent.pt"""
    )
    pdf.add_body(
        "The script opens an Isaac Sim window with the JetBot, green sphere, and ground "
        "plane. The trained policy runs in real-time - the JetBot actively tracks and "
        "reaches the sphere, which repositions after each reach."
    )
    pdf.add_body("Key differences from Isaac Lab evaluation:")
    pdf.add_bullet("Single environment only (no parallel envs)")
    pdf.add_bullet("Uses Isaac Sim WheeledRobot API instead of Isaac Lab Articulation")
    pdf.add_bullet("Green sphere is a USD prim, not a VisualizationMarker")
    pdf.add_bullet("No episode timeout - runs continuously until the window is closed")

    # ====================================================================
    # CHAPTER III: ISAAC SIM EXTENSION
    # ====================================================================
    pdf.add_chapter_title("Chapter III: Isaac Sim Extension - The Examples Browser")

    pdf.add_body(
        "The final step is packaging the policy deployment as a proper Isaac Sim "
        "extension that appears in the Examples Browser alongside NVIDIA's built-in "
        "examples (Hello World, Franka, Quadruped, etc.). This provides a polished "
        "UI with Load/Reset buttons, checkpoint configuration, and live status display."
    )

    # ---- 3.1 ----
    pdf.add_heading("3.1 Extension Architecture")

    pdf.add_body(
        "Isaac Sim extensions follow a well-defined architecture based on the Omniverse "
        "Kit framework. The key components are:"
    )
    pdf.add_bullet(
        "BaseSample: Abstract base class managing the World lifecycle (load, reset, clear). "
        "Provides hooks: setup_scene(), setup_post_load(), on_physics_step(), etc."
    )
    pdf.add_bullet(
        "BaseSampleUITemplate: UI framework with CollapsableFrames, Load/Reset buttons, "
        "and abstract methods for custom UI controls."
    )
    pdf.add_bullet(
        "Extension (omni.ext.IExt): Entry point that creates the sample + UI and registers "
        "with the Examples Browser."
    )

    pdf.add_body("Our extension consists of three files:")
    pdf.add_code_block(
        """sphere_follow/
  __init__.py                     # Exports
  sphere_follow.py                # SphereFollow(BaseSample)
  sphere_follow_extension.py      # SphereFollowUI + SphereFollowExtension"""
    )

    # ---- 3.2 ----
    pdf.add_heading("3.2 SphereFollow(BaseSample) - The Scene and Logic")

    pdf.add_body(
        "The SphereFollow class inherits from BaseSample and implements all lifecycle "
        "methods. The key pattern: scene setup is synchronous, post-load is async, and "
        "the policy runs in a physics callback."
    )

    pdf.add_subheading("Scene Setup")
    pdf.add_code_block(
        """class SphereFollow(BaseSample):
    def __init__(self):
        super().__init__()
        # Match training: 120Hz physics, 60Hz render
        self._world_settings["physics_dt"] = 1.0 / 120.0
        self._world_settings["rendering_dt"] = 1.0 / 60.0

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        # Spawn JetBot from USD asset
        self._jetbot = world.scene.add(WheeledRobot(
            prim_path="/World/Jetbot",
            wheel_dof_names=["left_wheel_joint",
                             "right_wheel_joint"],
            usd_path=jetbot_asset_path,
            position=np.array([0, 0, 0.05]),
        ))

        # Green sphere target (USD prim)
        self._sphere_prim = create_green_sphere(stage, ...)

        # Dome light matching training environment
        UsdLux.DomeLight.Define(stage, "/World/DomeLight")"""
    )

    pdf.add_subheading("Post-Load: Policy Loading and Physics Callback")
    pdf.add_code_block(
        """async def setup_post_load(self):
    # Load trained PPO checkpoint
    self._load_policy()

    # Register physics callback (runs at 120Hz)
    world.add_physics_callback("sphere_follow_step",
        callback_fn=self.on_physics_step)

    # Start simulation immediately
    await self.get_world().play_async()"""
    )

    pdf.add_subheading("Physics Step: Policy Inference at 60Hz")
    pdf.add_code_block(
        """def on_physics_step(self, step_size):
    # Decimation: policy every 2 physics steps
    self._decimation_counter += 1
    if self._decimation_counter % 2 != 0:
        return

    # Read state -> compute obs -> normalize -> policy -> apply
    robot_pos, robot_quat = self._jetbot.get_world_pose()
    obs = compute_observations(robot_pos, robot_quat,
        self._jetbot.get_linear_velocity(), self._sphere_pos)
    obs_norm = (torch.from_numpy(obs) - self._obs_mean) / self._obs_std

    with torch.no_grad():
        action = self._policy(obs_norm).squeeze().numpy()

    # Apply wheel velocity targets
    self._jetbot._articulation_view.set_joint_velocity_targets(...)

    # Check sphere reached -> reposition
    if dist < 0.3:
        self._sphere_pos = random_sphere_position(robot_pos)
        set_sphere_position(self._sphere_prim, self._sphere_pos)"""
    )

    # ---- 3.3 ----
    pdf.add_heading("3.3 The Extension UI")

    pdf.add_body(
        "The SphereFollowUI class extends BaseSampleUITemplate to add two extra frames:"
    )
    pdf.add_bullet(
        "Policy Configuration: A text field for the checkpoint path. The user can "
        "change this before clicking Load to point to a different checkpoint."
    )
    pdf.add_bullet(
        "Live Status: Real-time display of spheres reached, step count, 4D observations, "
        "and wheel velocity actions updated every ~0.5 seconds."
    )

    pdf.add_code_block(
        """class SphereFollowUI(BaseSampleUITemplate):
    def build_extra_frames(self):
        with self.extra_stacks:
            # Checkpoint path field
            checkpoint_frame = ui.CollapsableFrame(
                title="Policy Configuration", ...)
            with checkpoint_frame:
                self._checkpoint_field = ui.StringField()

            # Live status display
            status_frame = ui.CollapsableFrame(
                title="Live Status", ...)
            with status_frame:
                self._stats_label = ui.Label("Spheres: 0")
                self._obs_label = ui.Label("Obs: [...]")"""
    )

    pdf.add_subheading("Extension Registration")
    pdf.add_body(
        "The SphereFollowExtension registers with the Examples Browser so it appears "
        "in the Isaac Sim UI under the 'Policy' category:"
    )
    pdf.add_code_block(
        """class SphereFollowExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        sample = SphereFollow()
        ui_handle = SphereFollowUI(
            title="Wheeled Robot: JetBot Sphere Follow",
            sample=sample, ...)

        get_browser_instance().register_example(
            name="JetBot Sphere Follow",
            execute_entrypoint=ui_handle.build_window,
            ui_hook=ui_handle.build_ui,
            category="Policy",
        )"""
    )

    # ---- 3.4 ----
    pdf.add_heading("3.4 Installation and Registration")

    pdf.add_body(
        "To install the extension into Isaac Sim, copy the sphere_follow directory "
        "into the interactive examples extension and register it in extension.toml:"
    )
    pdf.add_code_block(
        """# File location:
isaacsim/exts/isaacsim.examples.interactive/
  isaacsim/examples/interactive/sphere_follow/
    __init__.py
    sphere_follow.py
    sphere_follow_extension.py

# Add to extension.toml:
[[python.module]]
name = "isaacsim.examples.interactive.sphere_follow" """
    )

    pdf.add_body(
        "After installation, the extension appears in Isaac Sim's Examples Browser. "
        "Navigate to Policy > JetBot Sphere Follow, click Load, and the trained "
        "JetBot will begin chasing the green sphere with the standard Load/Reset controls."
    )

    # ---- 3.5 ----
    pdf.add_heading("3.5 Comparing the Three Approaches")

    pdf.add_table(
        ["", "Isaac Lab", "Standalone", "Extension"],
        [
            ["Use case", "Training + eval", "Quick demo", "Integrated UI"],
            ["Framework", "Isaac Lab", "Isaac Sim only", "Isaac Sim + Kit"],
            ["Parallel envs", "Yes (100+)", "No (single)", "No (single)"],
            ["UI", "Terminal", "Viewport only", "Full UI panel"],
            ["Base class", "DirectRLEnv", "None (script)", "BaseSample"],
            ["Physics loop", "Env step()", "world.step()", "Physics callback"],
            ["Sphere", "VisualizationMarker", "USD prim", "USD prim"],
            ["Robot API", "Articulation", "WheeledRobot", "WheeledRobot"],
        ],
    )

    # ====================================================================
    # CONCLUSION
    # ====================================================================
    pdf.add_chapter_title("Conclusion")

    pdf.add_body(
        "We built a complete reinforcement learning pipeline for teaching a JetBot "
        "to chase a sphere, then deployed the trained policy in three different ways:"
    )
    pdf.add_bullet(
        "Isaac Lab: GPU-accelerated parallel training with 100 environments, "
        "training a working policy in ~10 minutes on the RTX PRO 6000 Blackwell."
    )
    pdf.add_bullet(
        "Isaac Sim Standalone: A self-contained script that loads the checkpoint "
        "and runs inference using only Isaac Sim APIs - no Isaac Lab required."
    )
    pdf.add_bullet(
        "Isaac Sim Extension: A fully integrated extension with UI controls, "
        "registered in the Examples Browser alongside NVIDIA's built-in samples."
    )

    pdf.add_body("Key takeaways:")
    pdf.add_bullet(
        "Environment design matters: compact observations (4D), shaped rewards "
        "(4 components), and proper episode structure are more important than "
        "network size or hyperparameter tuning."
    )
    pdf.add_bullet(
        "Parallel simulation is transformative: 100 environments on one GPU "
        "provide enough data to train a working policy in about 10 minutes."
    )
    pdf.add_bullet(
        "The Isaac Lab -> Isaac Sim pipeline is clean: the API mapping is "
        "straightforward, and the checkpoint contains everything needed for "
        "deployment (weights + observation normalizer)."
    )
    pdf.add_bullet(
        "The BaseSample/BaseSampleUITemplate pattern makes it easy to create "
        "professional Isaac Sim extensions with proper lifecycle management."
    )

    pdf.add_divider()

    pdf.add_subheading("Resources")
    pdf.add_bullet("NVIDIA Isaac Lab: github.com/isaac-sim/IsaacLab")
    pdf.add_bullet("NVIDIA Isaac Sim: docs.isaacsim.omniverse.nvidia.com")
    pdf.add_bullet("skrl RL Library: skrl.readthedocs.io")
    pdf.add_bullet("PPO Paper: Schulman et al., 2017 - arxiv.org/abs/1707.06347")
    pdf.add_bullet("Project Repository: github.com/kabilankb/isaaclab_jetbot")

    pdf.add_divider()
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(117, 117, 117)
    pdf.multi_cell(
        0, 6,
        "Built with Isaac Lab 0.47, Isaac Sim 5.1, skrl 1.4, and PyTorch. "
        "Trained on Dell Pro Max with NVIDIA RTX PRO 6000 Blackwell (96 GB)."
    )

    return pdf


if __name__ == "__main__":
    pdf = build_pdf()
    output_path = "/home/zeux/IsaacLabTutorial/JetBot_Sphere_Following_RL_Blog.pdf"
    pdf.output(output_path)
    print(f"PDF generated: {output_path}")

"""Generate a Medium-style blog post PDF about JetBot Sphere-Following with Isaac Lab."""

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

    # ---- Page 1: Title ----
    pdf.add_page()
    pdf.ln(15)
    pdf.add_title("Training a JetBot on Isaac Lab with the Dell Pro Max Powered by NVIDIA RTX PRO 6000 Blackwell")
    pdf.add_subtitle(
        "Building a reinforcement learning environment from scratch where a "
        "differential-drive robot learns to chase a moving sphere using PPO - "
        "trained in minutes on Blackwell-class GPU hardware."
    )
    pdf.add_author_line("February 2026  |  12 min read")
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
        "policy in roughly 10 minutes. In this post, we walk through the complete process "
        "of building this environment in NVIDIA Isaac Lab - from defining the robot and "
        "the target, to designing observations and rewards, to training a PPO policy and "
        "deploying it."
    )

    # ---- Section 1 ----
    pdf.add_heading("1. The Platform: Isaac Lab + JetBot")

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

    pdf.add_subheading("Environment Setup")
    pdf.add_body("The environment configuration defines the simulation parameters:")
    pdf.add_code_block(
        """@configclass
class SphereFollowEnvCfg(DirectRLEnvCfg):
    decimation = 2                    # control at 60 Hz
    episode_length_s = 20.0           # 20-second episodes
    action_space = 2                  # left/right wheel velocity
    observation_space = 4             # see Section 3
    env_spacing = 4.0                 # meters between envs
    sphere_radius = 0.1              # 10cm green sphere
    sphere_reach_threshold = 0.3     # "reached" at 30cm
    sphere_spawn_range_min = 0.5     # spawn 0.5-1.5m away
    sphere_spawn_range_max = 1.5"""
    )

    # ---- Section 2 ----
    pdf.add_heading("2. The Task: Sphere Following")
    pdf.add_body(
        "The task is conceptually simple: a green sphere spawns at a random position near "
        "the robot. The robot must navigate to the sphere. When it gets within 30cm, the "
        "sphere teleports to a new random location and the robot must chase it again. This "
        "cycle continues for the entire 20-second episode."
    )
    pdf.add_body(
        "The sphere is implemented as a visualization marker - it has no physics collider "
        "and no mass. Its position is controlled entirely by the environment code. This is "
        "intentional: we do not want the robot to push the sphere around, we want it to "
        "navigate to the sphere's location."
    )
    pdf.add_body("Each episode proceeds as follows:")
    pdf.add_bullet("Robot resets to its default pose at the center of the env cell")
    pdf.add_bullet("Sphere spawns at a random angle, 0.5-1.5m from the robot")
    pdf.add_bullet(
        "Robot takes actions (wheel velocities) based on observations"
    )
    pdf.add_bullet("When robot reaches sphere (dist < 0.3m): +5.0 bonus, sphere respawns")
    pdf.add_bullet("Episode ends after 20 seconds (timeout)")

    # ---- Section 3 ----
    pdf.add_heading("3. Observation Space Design")
    pdf.add_body(
        "Designing good observations is one of the most critical decisions in RL environment "
        "design. The observations must give the agent enough information to solve the task, "
        "but should be compact and normalized to help the neural network learn efficiently."
    )
    pdf.add_body("Our observation is a 4-dimensional vector computed every step:")

    pdf.add_code_block(
        """obs = [dot, cross_z, dist_norm, forward_speed]"""
    )

    pdf.add_bold_body("1. Dot Product (alignment signal)")
    pdf.add_body(
        "dot = sum(forward_dir * dir_to_sphere). This equals +1 when the robot faces "
        "directly toward the sphere, -1 when facing away, and 0 when perpendicular. "
        "It tells the agent HOW WELL it is aligned with the target."
    )

    pdf.add_bold_body("2. Cross Product Z-component (steering signal)")
    pdf.add_body(
        "cross_z = (forward x dir_to_sphere).z. This is positive when the sphere is "
        "to the left and negative when it is to the right. It tells the agent WHICH "
        "DIRECTION to turn. Combined with the dot product, these two signals fully "
        "encode the relative bearing to the sphere."
    )

    pdf.add_bold_body("3. Normalized Distance")
    pdf.add_body(
        "dist_norm = distance / 3.0, clamped to [0, 1]. This tells the agent HOW FAR "
        "the sphere is. We normalize by the maximum expected distance (3m) to keep "
        "values in a neural-network-friendly range."
    )

    pdf.add_bold_body("4. Forward Speed")
    pdf.add_body(
        "forward_speed = body-frame X velocity. This gives the agent proprioceptive "
        "feedback about its current motion, enabling it to learn smooth acceleration "
        "and deceleration behaviors."
    )

    pdf.add_quote(
        "These four numbers are sufficient for the agent to solve the task. "
        "We deliberately avoided raw positions or quaternions - the ego-centric "
        "representation generalizes better across environments."
    )

    # ---- Section 4 ----
    pdf.add_heading("4. Reward Function Engineering")
    pdf.add_body(
        "The reward function is the language through which we communicate our "
        "intentions to the RL agent. A well-designed reward makes the difference "
        "between an agent that learns in minutes and one that never converges."
    )
    pdf.add_body("Our reward is a sum of four components:")

    pdf.add_code_block(
        """reward = approach + alignment + reach_bonus + time_penalty"""
    )

    pdf.add_subheading("Approach Reward (dense, distance-based)")
    pdf.add_code_block(
        """approach = (prev_dist - curr_dist) * 1.0"""
    )
    pdf.add_body(
        "This is a potential-based shaping reward. Every step, we compare the "
        "current distance to the previous distance. If the robot moved closer, "
        "the reward is positive. If it moved away, negative. This gives a dense, "
        "continuous learning signal at every timestep."
    )

    pdf.add_subheading("Alignment Reward (orientation incentive)")
    pdf.add_code_block(
        """alignment = dot(forward, dir_to_sphere) * 0.5"""
    )
    pdf.add_body(
        "This rewards the robot for facing the sphere, even before it starts "
        "moving. It encourages the agent to first turn toward the target, then "
        "drive forward - a natural and efficient navigation strategy."
    )

    pdf.add_subheading("Reach Bonus (sparse, goal-based)")
    pdf.add_code_block(
        """reach = 5.0 if dist < 0.3m else 0.0"""
    )
    pdf.add_body(
        "A one-time bonus when the robot reaches the sphere. After collecting "
        "the bonus, the sphere respawns at a new random location. The agent "
        "learns that reaching spheres is highly valuable, creating the "
        "continuous chasing behavior we want."
    )

    pdf.add_subheading("Time Penalty (efficiency pressure)")
    pdf.add_code_block(
        """time_penalty = -0.01 per step"""
    )
    pdf.add_body(
        "A small constant penalty at every step. This discourages the agent "
        "from spinning in circles or sitting still. Combined with the reach "
        "bonus, it creates pressure to reach spheres as quickly as possible."
    )

    # ---- Section 5 ----
    pdf.add_heading("5. The Control System: From Observations to Wheel Velocities")

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
7. Compute reward, check termination
8. Repeat"""
    )

    pdf.add_subheading("How Differential Drive Creates Motion")
    pdf.add_body(
        "The differential drive kinematics are elegant in their simplicity:"
    )
    pdf.add_bullet("Both wheels same speed: robot drives straight forward")
    pdf.add_bullet("Left wheel faster: robot curves right")
    pdf.add_bullet("Right wheel faster: robot curves left")
    pdf.add_bullet("Wheels opposite directions: robot spins in place")
    pdf.add_body(
        "The RL agent discovers these relationships purely through trial and error. "
        "It is never told the kinematics equations - it learns them implicitly by "
        "observing the consequences of its actions."
    )

    pdf.add_subheading("Network Architecture")
    pdf.add_body("The policy and value networks share a common backbone:")
    pdf.add_code_block(
        """Input:  4 observations
  |
  v
Linear(4 -> 64) + ELU
Linear(64 -> 64) + ELU
  |            |
  v            v
Policy:      Value:
Linear(64->2) Linear(64->1)
  |              |
  v              v
[left_vel,    state value
 right_vel]   estimate"""
    )
    pdf.add_body(
        "The shared backbone (two 64-unit layers with ELU activations) extracts "
        "features from the observation. The policy head outputs two continuous "
        "values (wheel velocities). The value head estimates the expected future "
        "return, which is used by PPO for advantage estimation."
    )

    # ---- Section 6 ----
    pdf.add_heading("6. Training with PPO")

    pdf.add_subheading("Why PPO?")
    pdf.add_body(
        "Proximal Policy Optimization (PPO) is the workhorse of modern robot RL. "
        "It is stable, sample-efficient with parallel environments, and works well "
        "with continuous action spaces. Key properties:"
    )
    pdf.add_bullet("Clipped surrogate objective prevents destructively large policy updates")
    pdf.add_bullet("Generalized Advantage Estimation (GAE) balances bias and variance")
    pdf.add_bullet("Works with shared policy-value networks")
    pdf.add_bullet("Scales linearly with number of parallel environments")

    pdf.add_subheading("Training Configuration")

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
        "Training proceeds through recognizable phases:"
    )
    pdf.add_body(
        "Phase 1 - Random exploration: The agent outputs near-random wheel velocities. "
        "Occasionally it stumbles toward a sphere by accident and receives the reach bonus. "
        "The approach reward provides gradient signal even during random motion."
    )
    pdf.add_body(
        "Phase 2 - Turning behavior emerges: The agent learns that facing the sphere "
        "(high dot product) yields positive alignment reward. It begins to reliably turn "
        "toward the target."
    )
    pdf.add_body(
        "Phase 3 - Drive-and-chase: The agent combines turning with forward motion. "
        "It reaches spheres more frequently, collecting more reach bonuses. The time "
        "penalty pushes it to reach spheres faster."
    )
    pdf.add_body(
        "Phase 4 - Refined tracking: The agent learns smooth, efficient trajectories. "
        "It anticipates the turn needed and drives curved paths directly to the target "
        "rather than stop-turn-drive sequences."
    )

    # ---- Section 7 ----
    pdf.add_heading("7. Checkpoint Policy and Deployment")

    pdf.add_body(
        "During training, skrl automatically saves checkpoints at regular intervals "
        "and tracks the best-performing agent:"
    )

    pdf.add_code_block(
        """logs/skrl/sphere_follow_direct/
  2026-02-21_19-46-44_ppo_torch/
    checkpoints/
      agent_2400.pt      # checkpoint at 2400 steps
      agent_4800.pt      # checkpoint at 4800 steps
      ...
      agent_24000.pt     # final checkpoint
      best_agent.pt      # best by episode return
    params/
      env.yaml           # environment config
      agent.yaml         # agent hyperparameters"""
    )

    pdf.add_subheading("Loading and Evaluating a Policy")
    pdf.add_body("To evaluate the trained agent:")
    pdf.add_code_block(
        """./launch.sh play \\
    --checkpoint logs/skrl/.../best_agent.pt \\
    --num_envs 10"""
    )
    pdf.add_body(
        "This loads the checkpoint weights into the same network architecture, "
        "then runs the environment with the learned policy. The agent acts "
        "deterministically (no exploration noise) during evaluation."
    )

    pdf.add_subheading("What the Checkpoint Contains")
    pdf.add_body("Each .pt file is a PyTorch state dict containing:")
    pdf.add_bullet("Policy network weights and biases (4 -> 64 -> 64 -> 2)")
    pdf.add_bullet("Value network weights (shared backbone + value head)")
    pdf.add_bullet("Log standard deviation parameters for the Gaussian policy")
    pdf.add_bullet("Optimizer state (for resuming training)")
    pdf.add_bullet("Running mean/std for observation and value normalization")

    # ---- Section 8 ----
    pdf.add_heading("8. Key Implementation Details")

    pdf.add_subheading("Sphere Repositioning Without Reward Spikes")
    pdf.add_body(
        "When the robot reaches a sphere, we reposition it to a new random location. "
        "A naive implementation would cause a reward spike on the next step: "
        "prev_dist is small (near the old sphere) but curr_dist is large (far from "
        "the new sphere), yielding a huge negative approach reward."
    )
    pdf.add_body("Our solution: after repositioning, we immediately reset prev_dist:")
    pdf.add_code_block(
        """reached_ids = torch.where(curr_dist < 0.3)[0]
if len(reached_ids) > 0:
    self._spawn_sphere_positions(reached_ids)
    # Reset prev_dist to avoid reward spike
    new_diff = sphere_pos[reached_ids] - robot_pos[reached_ids]
    curr_dist[reached_ids] = torch.linalg.norm(new_diff)
self.prev_dist_to_sphere = curr_dist.clone()"""
    )

    pdf.add_subheading("Stale Simulation Data During Reset")
    pdf.add_body(
        "A subtle but critical bug: after calling write_root_state_to_sim(), the "
        "robot's position data is not updated until the next simulation step. Reading "
        "root_pos_w immediately returns stale values. We solve this by passing the "
        "known reset position directly:"
    )
    pdf.add_code_block(
        """# Use known reset position, not stale sim data
robot_reset_pos_xy = default_root_state[:, :2]
self._spawn_sphere_positions(env_ids,
    robot_pos_xy=robot_reset_pos_xy)"""
    )

    pdf.add_subheading("Ego-Centric Observations")
    pdf.add_body(
        "We use ego-centric (robot-relative) observations rather than world-frame "
        "coordinates. The dot product and cross product encode the relative bearing "
        "to the sphere in the robot's reference frame. This means the policy "
        "generalizes across all environment positions - it does not overfit to "
        "absolute coordinates."
    )

    # ---- Section 9 ----
    pdf.add_heading("9. Launch Commands: The Complete Workflow")

    pdf.add_body(
        "One of the strengths of this project is the single-entry-point launch script "
        "that wraps all operations. Here is the full workflow from installation to "
        "evaluation, with every command you need."
    )

    pdf.add_subheading("Step 1: Install the Package")
    pdf.add_body(
        "Install the isaac_lab_tutorial package in editable mode. This registers "
        "the environments with gymnasium so Isaac Lab can discover them."
    )
    pdf.add_code_block(
        """./launch.sh install

# Or using Make:
make install"""
    )

    pdf.add_subheading("Step 2: Verify Environment Registration")
    pdf.add_body(
        "Confirm both the original direction-following and the new sphere-following "
        "environments are registered."
    )
    pdf.add_code_block(
        """./launch.sh list-envs

# Expected output:
# - Template-Isaac-Lab-Tutorial-Direct-v0
# - Isaac-Lab-Tutorial-SphereFollow-Direct-v0"""
    )

    pdf.add_subheading("Step 3: Visual Smoke Test with Random Agent")
    pdf.add_body(
        "Run the environment with random actions to verify the simulation works, "
        "green spheres are visible, and environments run without errors. The robot "
        "will move erratically - this is expected with random actions."
    )
    pdf.add_code_block(
        """./launch.sh random-agent --num_envs 10

# Or with zero actions (robot stays still, spheres visible):
./launch.sh zero-agent --num_envs 10

# Using Make:
make random-agent NUM_ENVS=10"""
    )

    pdf.add_subheading("Step 4: Train the PPO Agent")
    pdf.add_body(
        "Launch training with PPO across 100 parallel environments. Training runs "
        "for 24,000 timesteps and saves checkpoints to the logs directory."
    )
    pdf.add_code_block(
        """# Default training (100 envs, PPO, 24000 timesteps):
./launch.sh train --algorithm PPO --num_envs 100

# With custom iterations (e.g., 500 updates):
./launch.sh train --algorithm PPO --num_envs 100 \\
    --max_iterations 500

# With a specific seed for reproducibility:
./launch.sh train --algorithm PPO --num_envs 100 \\
    --seed 42

# Resume training from a checkpoint:
./launch.sh train --algorithm PPO --num_envs 100 \\
    --checkpoint logs/skrl/sphere_follow_direct/\\
    <run_dir>/checkpoints/agent_24000.pt

# Using Make:
make train ALGORITHM=PPO NUM_ENVS=100"""
    )

    pdf.add_subheading("Step 5: Evaluate the Trained Agent")
    pdf.add_body(
        "Load the best checkpoint and watch the JetBot actively chase the sphere. "
        "The robot will turn toward the sphere, drive to it, and when the sphere "
        "repositions, immediately begin tracking the new location."
    )
    pdf.add_code_block(
        """# Play the best agent:
./launch.sh play --checkpoint \\
    logs/skrl/sphere_follow_direct/\\
    <run_dir>/checkpoints/best_agent.pt

# With fewer envs for clearer visualization:
./launch.sh play --checkpoint \\
    logs/skrl/sphere_follow_direct/\\
    <run_dir>/checkpoints/best_agent.pt \\
    --num_envs 10

# Using Make:
make play CHECKPOINT=logs/skrl/sphere_follow_direct/\\
    <run_dir>/checkpoints/best_agent.pt"""
    )

    pdf.add_subheading("Step 6: Record Video (Optional)")
    pdf.add_body(
        "Capture training or evaluation videos for documentation or sharing."
    )
    pdf.add_code_block(
        """# Record during training:
./launch.sh train --algorithm PPO --num_envs 100 \\
    --video

# Record during evaluation:
./launch.sh play --checkpoint <path> --video"""
    )

    pdf.add_subheading("Quick Reference: All Commands")
    pdf.add_code_block(
        """./launch.sh install          # Install package
./launch.sh list-envs        # List registered envs
./launch.sh check            # Verify prerequisites
./launch.sh random-agent     # Run with random actions
./launch.sh zero-agent       # Run with zero actions
./launch.sh train            # Train RL agent
./launch.sh play             # Evaluate trained agent
./launch.sh help             # Show all options"""
    )

    pdf.add_subheading("Make Targets")
    pdf.add_code_block(
        """make install       # Install package
make list-envs     # List environments
make check         # Verify prerequisites
make random-agent  # Random actions (NUM_ENVS=10)
make zero-agent    # Zero actions
make train         # Train (ALGORITHM=PPO NUM_ENVS=100)
make train-ppo     # Shortcut for PPO training
make play          # Evaluate (CHECKPOINT=<path>)"""
    )

    # ---- Section 10 ----
    pdf.add_heading("10. Conclusion")
    pdf.add_body(
        "We built a complete reinforcement learning pipeline for teaching a JetBot "
        "to chase a sphere in NVIDIA Isaac Lab. The key takeaways:"
    )
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
        "Differential drive is deceptively rich: two wheel velocities produce "
        "complex emergent behaviors when optimized by RL - smooth curves, "
        "spin-and-drive, and efficient tracking trajectories."
    )
    pdf.add_bullet(
        "Implementation details matter: stale simulation data, reward spikes "
        "during respawning, and observation normalization can make or break "
        "training convergence."
    )
    pdf.add_body(
        "This environment serves as a foundation for more complex tasks: "
        "multi-target tracking, obstacle avoidance, sim-to-real transfer to a "
        "physical JetBot, or curriculum learning where the sphere spawns "
        "progressively farther away. The same framework scales to quadrupeds, "
        "manipulators, and humanoids - the patterns remain the same."
    )

    pdf.add_divider()

    pdf.add_subheading("Resources")
    pdf.add_bullet("NVIDIA Isaac Lab: github.com/isaac-sim/IsaacLab")
    pdf.add_bullet("skrl RL Library: skrl.readthedocs.io")
    pdf.add_bullet("PPO Paper: Schulman et al., 2017 - arxiv.org/abs/1707.06347")

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

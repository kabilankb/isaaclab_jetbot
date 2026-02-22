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

import os

import omni.ext
import omni.ui as ui
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate
from isaacsim.examples.interactive.sphere_follow import SphereFollow
from isaacsim.gui.components.ui_utils import get_style


class SphereFollowUI(BaseSampleUITemplate):
    """UI handler for JetBot Sphere-Following example with extra controls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._status_label = None
        self._stats_label = None
        self._obs_label = None
        self._checkpoint_field = None

    def build_extra_frames(self):
        with self.extra_stacks:
            # Checkpoint configuration frame
            checkpoint_frame = ui.CollapsableFrame(
                title="Policy Configuration",
                width=ui.Fraction(1),
                height=0,
                collapsed=False,
                style=get_style(),
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            )
            with checkpoint_frame:
                with ui.VStack(style=get_style(), spacing=5, height=0):
                    ui.Label("Checkpoint Path:", height=18)
                    self._checkpoint_field = ui.StringField(height=22)
                    self._checkpoint_field.model.set_value(SphereFollow.DEFAULT_CHECKPOINT)
                    ui.Label(
                        "Set the path before clicking 'Load'.",
                        style={"color": 0xFF888888, "font_size": 12},
                        height=16,
                    )

            # Live status frame
            status_frame = ui.CollapsableFrame(
                title="Live Status",
                width=ui.Fraction(1),
                height=0,
                collapsed=False,
                style=get_style(),
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            )
            with status_frame:
                with ui.VStack(style=get_style(), spacing=4, height=0):
                    self._status_label = ui.Label(
                        "Status: Ready - Press 'Load' to begin",
                        height=20,
                    )
                    self._stats_label = ui.Label(
                        "Spheres reached: 0  |  Steps: 0",
                        height=20,
                    )
                    self._obs_label = ui.Label(
                        "Obs: [-, -, -, -]  Act: [-, -]",
                        height=20,
                    )

    def _on_status_update(self, spheres_reached, step_count, obs, action):
        """Callback from SphereFollow sample to update UI labels."""
        if self._stats_label:
            self._stats_label.text = f"Spheres reached: {spheres_reached}  |  Steps: {step_count}"
        if self._obs_label and obs is not None and action is not None:
            self._obs_label.text = (
                f"Obs: [{obs[0]:+.2f}, {obs[1]:+.2f}, {obs[2]:.2f}, {obs[3]:+.2f}]  "
                f"Act: [{action[0]:+.3f}, {action[1]:+.3f}]"
            )

    def _on_load_world(self):
        # Set checkpoint path BEFORE load_world_async runs setup_post_load
        if self._checkpoint_field:
            path = self._checkpoint_field.model.get_value_as_string()
            self._sample.set_checkpoint_path(path)
        # Register status callback before load
        self._sample.set_status_callback(self._on_status_update)
        # Call parent to trigger the async load
        super()._on_load_world()

    def post_load_button_event(self):
        if self._status_label:
            self._status_label.text = "Status: World loaded - Running policy"

    def post_reset_button_event(self):
        if self._status_label:
            self._status_label.text = "Status: Reset complete - Running policy"
        if self._stats_label:
            self._stats_label.text = "Spheres reached: 0  |  Steps: 0"
        if self._obs_label:
            self._obs_label.text = "Obs: [-, -, -, -]  Act: [-, -]"

    def post_clear_button_event(self):
        if self._status_label:
            self._status_label.text = "Status: Cleared - Press 'Load' to begin"
        if self._stats_label:
            self._stats_label.text = "Spheres reached: 0  |  Steps: 0"
        if self._obs_label:
            self._obs_label.text = "Obs: [-, -, -, -]  Act: [-, -]"


class SphereFollowExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.example_name = "JetBot Sphere Follow"
        self.category = "Policy"

        overview = "This example demonstrates a JetBot differential-drive robot "
        overview += "following a green sphere target using a PPO policy trained in Isaac Lab. "
        overview += "The sphere repositions when the JetBot reaches it, creating a continuous "
        overview += "target-tracking task. The policy uses 4D ego-centric observations "
        overview += "(alignment, steering signal, distance, forward speed) to output "
        overview += "left/right wheel velocity commands."

        sample = SphereFollow()

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Wheeled Robot: JetBot Sphere Follow",
            "doc_link": "https://docs.isaacsim.omniverse.nvidia.com/latest/isaac_lab_tutorials/tutorial_policy_deployment.html",
            "overview": overview,
            "sample": sample,
        }

        ui_handle = SphereFollowUI(**ui_kwargs)

        # Register with the examples browser
        get_browser_instance().register_example(
            name=self.example_name,
            execute_entrypoint=ui_handle.build_window,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )

        return

    def on_shutdown(self):
        get_browser_instance().deregister_example(name=self.example_name, category=self.category)
        return

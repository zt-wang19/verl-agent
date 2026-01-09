# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------- Jericho (Interactive Fiction) --------------------- #

JERICHO_TEMPLATE_NO_HIS = """
You are playing an interactive fiction (text adventure) game called "{game_name}".
Your goal is: {task_description}

Your current observation is:
{current_observation}

Your current inventory: {inventory}
Your current score: {score}/{max_score}

Available actions (suggestions): [{available_actions}]

Now it's your turn to take an action.
You should first reason step-by-step about the current situation and what you should do next. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an action for the current step and present it within <action> </action> tags.
Note: You can also try other commands not in the suggested list, such as 'north', 'south', 'take [object]', 'examine [object]', 'open [object]', etc.
"""

JERICHO_TEMPLATE = """
You are playing an interactive fiction (text adventure) game called "{game_name}".
Your goal is: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took:
{action_history}

You are now at step {current_step} and your current observation is:
{current_observation}

Your current inventory: {inventory}
Your current score: {score}/{max_score}

Available actions (suggestions): [{available_actions}]

Now it's your turn to take an action.
You should first reason step-by-step about the current situation and what you should do next. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an action for the current step and present it within <action> </action> tags.
Note: You can also try other commands not in the suggested list, such as 'north', 'south', 'take [object]', 'examine [object]', 'open [object]', etc.
"""

JERICHO_TEMPLATE_SIMPLE = """
You are playing "{game_name}", an interactive fiction game.
Goal: {task_description}

Current observation: {current_observation}
Inventory: {inventory}
Score: {score}/{max_score}
Suggested actions: [{available_actions}]

Think about what to do next within <think></think> tags, then output your action within <action></action> tags.
"""




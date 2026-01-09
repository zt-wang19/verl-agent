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

from typing import List, Tuple
import re


def textworld_express_projection(
    actions: List[str],
    action_pools: List[List[str]] = None
) -> Tuple[List[str], List[int]]:
    """
    Process actions from LLM output to TextWorldExpress environment actions.
    
    Args:
        actions: List of raw action strings from LLM
        action_pools: Optional list of valid action pools for each environment
        
    Returns:
        processed_actions: List of processed action strings
        valids: List of validity flags (1 = valid, 0 = invalid)
    """
    valids = [0] * len(actions)
    processed_actions = []
    
    for i in range(len(actions)):
        original_str = actions[i]
        action_str = actions[i].lower()
        
        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = action_str.find(start_tag)
        end_idx = action_str.find(end_tag)
        
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block
                # Try to extract a reasonable action from the last part
                processed_actions.append(action_str[-50:].strip())
                continue
            
            # Extract just the content between the tags
            extracted_action = action_str[start_idx + len(start_tag):end_idx].strip()
            
            # Clean up the action string
            extracted_action = extracted_action.strip()
            
            processed_actions.append(extracted_action)
            valids[i] = 1
            
        except Exception:
            processed_actions.append(action_str[-50:].strip())
        
        # Check <think>...</think> - require reasoning
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0
        
        # Check if contains any Chinese characters (invalid for English environment)
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0
    
    return processed_actions, valids


import os
import pytest
import json
import shutil
from envs.pcgrl_env import get_available_tiles
from pcgrllm.generate_reward import generate_reward
from conf.config import TrainConfig, TrainLLMConfig
from debug.scenario_levels import AllLevels
from debug.validate_dungeon3 import validate_dungeon3
from pcgrllm.paths import init_config
from pcgrllm.scenario_preset import ScenarioPreset


# Copy ../pcgrllm/prompt to current directory using library
shutil.copytree('../pcgrllm/prompt', 'prompt', dirs_exist_ok=True)


def generate_and_save_reward(train_config, iteration):
    """
    Generates a reward function and saves it to a file.

    Parameters:
        train_config: The training configuration object.
        iteration: The current iteration for generating the reward.

    Returns:
        The path of the saved reward file.
    """
    gpt_model = ''
    # Create a unique reward directory for this test
    reward_dir = f"reward_outputs_{iteration}"
    # Remove the reward_dir if it exists to ensure a clean slate
    if os.path.exists(reward_dir):
        shutil.rmtree(reward_dir)
    os.makedirs(reward_dir, exist_ok=True)

    # Define the target character or scenario
    target_character = train_config.target_character
    if train_config.task == 'scenario' and target_character.isnumeric():
        scenario_preset = ScenarioPreset()
        scenario = scenario_preset.scenarios.get(target_character, None)
        target_character = scenario.prompt if scenario else target_character

    # Generate reward arguments
    args_dict = {
        'shared_storage_path': os.getcwd(),
        'postfix': f"reward_outer_{iteration}",
        'reward_functions_dir': reward_dir,
        'gpt_model': train_config.gpt_model,
        'gpt_max_token': 4096,
        'previous_reward_function': None,
        'trial_count': train_config.n_generation_trials,
        'total_iterations': train_config.total_iterations,
        'n_inner': 1,
        'iteration_num': iteration,
        'target_character': target_character,
        'pe': train_config.pe,
        'branch_factor': train_config.branch_factor,
        'feedback_path': None,
        'map_width': train_config.map_width,
        'map_height': train_config.map_width,
        'feature': train_config.reward_feature,
        'available_tiles': get_available_tiles(train_config.problem),
        'prev_eval_result': None,
        'auxiliary_prompt_path': None,
        'n_codegen_trials': train_config.n_codegen_trials,
        'n_codefix_trials': train_config.n_codefix_trials,
        'task': train_config.task,
    }

    result = generate_reward(train_config, args_dict, return_error=True)

    print("Result: ", result)

    assert result is not tuple  # Ensure the reward generation was successful

    return os.path.join(reward_dir, result)  # Return the full path of the reward file


def validate_generated_reward(train_config, reward_path):
    """
    Validates the dungeon with the generated reward function.

    Parameters:
        train_config: The training configuration object.
        reward_path: Path to the generated reward file.
    """
    # Run the validation
    train_config.reward_function_path = reward_path
    is_valid = validate_dungeon3(train_config)

    # Report validation result
    if is_valid:
        print(f"Validation successful for reward file: {reward_path}")
    else:
        print(f"Validation failed for reward file: {reward_path}")

    return is_valid


@pytest.mark.parametrize("index", list(range(1, 6)))
def test_generate_and_validate(index):
    """
    Tests reward generation and validation for all levels in AllLevels.

    Parameters:
        index: Index of the level in AllLevels.
    """
    train_config = TrainLLMConfig()
    init_config(train_config)

    iteration = index  # Use the index as the iteration number

    # Generate and save reward
    reward_path = generate_and_save_reward(train_config, iteration)

    # Validate the generated reward
    is_valid = validate_generated_reward(train_config, reward_path)

    # Assert validation result
    assert is_valid, f"Validation failed for level at index: {index}"

    print(f"Test passed for level index: {index}")

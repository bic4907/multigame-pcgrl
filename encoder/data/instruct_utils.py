from typing import Any
import os
from conf.config import Config
import json
from copy import deepcopy
from transformers import (
    FlaxBertModel,
    FlaxRobertaModel,
    FlaxAlbertModel,
    FlaxElectraModel,
    AutoTokenizer,
)

# from conf.config import TrainLLMConfig


def BERT(config):
    model_name = f"bert-{config.model_size}-uncased"
    model = FlaxBertModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def read_file(file_path: str) -> Any:
    """Reads and returns the content of a file."""
    with open(file_path, "r") as f:
        return f.read()


def get_ckpt_dir(config: Config):
    """Function to retrieve the checkpoint save path"""
    return os.path.join(config.exp_dir, "ckpts")


def apply_pretrained_model(config):

    if config.pretrained_model == "bert":
        model, tokenizer = BERT(config)
    elif config.pretrained_model == "roberta":
        model, tokenizer = Roberta(config)
    elif config.pretrained_model == "albert":
        model, tokenizer = Albert(config)
    elif config.pretrained_model == "electra":
        model, tokenizer = Electra(config)
    else:
        raise ValueError(f"Model {config.pretrained_model} not supported")
    return model, tokenizer



def prompt_combination(
        data, scenario_combination, current_prompt, evaluation_values, num_scenario, similar, use_ai
):
    """
    prompt combination

    Args:
        data (_type_): _description_
        scenario_combination (_type_): _description_
        current_prompt (_type_): _description_
        evaluation_values (_type_): _description_
        num_scenario (_type_): _description_

    Returns:
        _type_: _description_
    """

    if not scenario_combination:
        lsttext = current_prompt + "."
        return {lsttext: evaluation_values}

    final_prompts = {}
    scenario_key = scenario_combination.popleft()

    scenario = data["scenarios"].get(scenario_key)
    if not scenario:
        return final_prompts

    prompt = scenario.get("prompt", "")
    feature = scenario.get("feature", "")

    for key, value in data[feature].items():
        if similar:
            similar_words = value["similar"]
        else:
            similar_words = [value["similar"][0]]

        for similar_word in similar_words:
            formatted_prompt = prompt.format(**{feature: similar_word})
            if current_prompt == "":
                updated_prompt = formatted_prompt
            else:
                updated_prompt = current_prompt + ", " + formatted_prompt

            updated_evaluation_values = deepcopy(evaluation_values)

            if value["sub_condition"] == "AI":
                continue

            updated_evaluation_values[feature] = [value["value"], value["sub_condition"]]
            sub_prompts = prompt_combination(
                data=data,
                scenario_combination=scenario_combination.copy(),
                current_prompt=updated_prompt,
                evaluation_values=updated_evaluation_values,
                num_scenario=num_scenario,
                similar=similar,
                use_ai=use_ai,
            )
            if sub_prompts and len(list(sub_prompts.values())[0]) == num_scenario:
                final_prompts.update(sub_prompts)

    return final_prompts


def match_prompt_eval(config, combinations):
    """_summary_

    Args:
        config (_type_): _description_

    Returns:
        _type_: _description_
    """

    with open(config.prompt_path, "r") as file:
        data = json.load(file)

    text = ""
    eval_list = {}

    from collections import deque

    prompt_list = {}
    prompt_list.update(
        prompt_combination(
            data=data,
            scenario_combination=deque(combinations),
            current_prompt=text,
            evaluation_values=eval_list,
            num_scenario=config.num_scenario,
            similar=config.similar_words,
            use_ai=config.use_ai,
        )
    )

    return prompt_list

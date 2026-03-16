from enum import IntEnum
from encoder.data.instruct_utils import *
import time
import jax
import pandas as pd
import random
import numpy as np
import itertools


class ConditionFeature(IntEnum):
    region = 1
    path_length = 2
    block = 3
    bat_amount = 4
    bat_direction = 5


def dict_to_csv(prompt_eval):
    instructs = list(prompt_eval.keys())
    conditions = []
    conditions_values = []
    conditions_sub_values = []
    example_values = [-1] * 9

    for v in prompt_eval.values():
        condition, values_sub = list(v.keys()), list(v.values())
        values = [v_s[0] for v_s in values_sub]
        sub_cond = [v_s[1] for v_s in values_sub]
        values = [int(x) if str(x).isdigit() else x for x in values]

        condition_str = "".join(sorted([str(ConditionFeature[cond]) for cond in condition], key=int))
        example_value = deepcopy(example_values)
        example_sub_value = deepcopy(example_values)

        for idx in range(len(condition)):
            example_value[ConditionFeature[condition[idx]] - 1] = values[idx]
            example_sub_value[ConditionFeature[condition[idx]] - 1] = sub_cond[idx]

        conditions.append(int(condition_str))
        conditions_values.append(example_value)
        conditions_sub_values.append(example_sub_value)


    conditions = np.array(conditions)
    conditions_values = np.array(conditions_values)
    conditions_sub_values = np.array(conditions_sub_values)

    prompt_eval_csv = pd.DataFrame(instructs, columns=["instruction"])
    prompt_eval_csv["reward_enum"] = conditions
    for i in range(conditions_values.shape[1]):
        prompt_eval_csv[f"condition_{i}"] = conditions_values[:, i]
        prompt_eval_csv[f"sub_condition_{i}"] = conditions_sub_values[:, i]

    return prompt_eval_csv


def make_data(config, random_list, combination, pretrained_model, tokenizer):

    prompt_eval = match_prompt_eval(config, combination)
    prompt_eval_csv = dict_to_csv(prompt_eval)

    text_samples = list(prompt_eval_csv["instruction"])



    with jax.disable_jit():  # Ensure BERT is not traced/optimized during training
        # Tokenize and preprocess
        input_text = text_samples
        # jax.debug.print(f"{input_text}")
        start_time = time.time()
        encoded_inputs = tokenizer(
            input_text,
            return_tensors="jax",
            padding="max_length",
            max_length=config.max_length,
            truncation=True,
        )
        end_time = time.time()
        jax.debug.print(f"tokenizer: {end_time - start_time:.4f} sec")
        outputs = pretrained_model(**encoded_inputs).last_hidden_state
        end_time2 = time.time()
        jax.debug.print(f"pretrained_model: {end_time2 - end_time:.4f} sec")
        jax.debug.print(f"whole_process: {end_time2 - start_time:.4f} sec")
        jax.debug.print(f"{outputs.shape}")
        cls_outputs = outputs[:, 0, :]  # Take the [CLS] token outputs
        jax.debug.print(f"{cls_outputs.shape}")

    if random_list is None:
        num_train = int(len(prompt_eval_csv) * config.train_ratio)
        num_test = len(prompt_eval_csv) - num_train

        random_list = [True] * num_train + [False] * num_test
        random.shuffle(random_list)

    prompt_eval_csv["train"] = random_list

    embedding_df = pd.DataFrame(
        cls_outputs, columns=[f"embed_{i + 1}" for i in range(cls_outputs.shape[1])]
    )
    prompt_eval_csv = pd.concat([prompt_eval_csv, embedding_df], axis=1)

    return prompt_eval_csv, random_list


def main():
    config = TrainLLMConfig()
    models = ["bert"]
    config.start_scenario = 1
    config.end_scenario = 5
    config.num_scenario = 1

    config.similar_words = True
    config.use_ai = False
    random_list = None

    for model_name in models:
        config.pretrained_model = model_name
        print(config.pretrained_model)

        scenario_keys = [
            str(i + 1) for i in range(config.start_scenario - 1, config.end_scenario)
        ]
        if config.permutation:
            combinations = list(itertools.permutations(scenario_keys, config.num_scenario))
        else:
            combinations = list(itertools.combinations(scenario_keys, config.num_scenario))

        pretrained_model, tokenizer = apply_pretrained_model(config)

        for combination in combinations:

            print(f"combination: {combination}")

            prompt_eval_csv, random_list = make_data(config, random_list, combination, pretrained_model, tokenizer)

            aug = ""
            if config.similar_words:
                aug += "sw"
            if config.permutation:
                aug += "pm"
            aug = "false" if aug == "" else aug

            os.makedirs(f"instruct/aug_{aug}/{config.pretrained_model}", exist_ok=True)

            prompt_eval_csv.to_csv(
                f"instruct/aug_{aug}/{config.pretrained_model}/scn-{config.num_scenario}_se-{''.join(sorted(combination, key=int))}.csv",
                index=False,
            )


if __name__ == "__main__":
    main()
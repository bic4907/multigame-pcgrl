import argparse
import pickle
from functools import partial
from os.path import abspath, join, dirname
import pandas as pd
from tabulate import tabulate

from instruct_rl.evaluation.evaluator import CosineEvaluator


if __name__ == "__main__":


    instruction = abspath(join(__file__, "..", "..", "..", "instruct", "test", "bert", "scn-1_se-whole.csv"))
    default_dest_dir = abspath(join(dirname(__file__), "level_dataset"))

    parser = argparse.ArgumentParser(description="Copy evaluation data")
    parser.add_argument('--src', type=str, required=True, help='Source directory to copy from')
    parser.add_argument('--render_dest', type=str, default=default_dest_dir, help='Destination directory to copy to')
    parser.add_argument('--instruction', type=str, default=instruction)

    args = parser.parse_args()

    with open(args.instruction, "r") as f:
        inst_df = pd.read_csv(f)

    """
    1. Creating the Image Dataset to Pickle 
    """
    # dataset = get_dataset(args.src, inst_df=inst_df)
    # print(f"Dataset loaded with {len(dataset)} levels.")
    #
    # human_level_df, human_images = get_valid_dataset(dataset, 'human')
    # rl_level_df, rl_images = get_valid_dataset(dataset, 'rl', n=300)
    # gpt_level_df, gpt_images = get_valid_dataset(dataset, 'gpt', n=300)
    #
    # print(f"Human images shape: {human_images.shape}, {len(human_level_df)}")
    # print(f"RL images shape: {rl_images.shape}, {len(rl_level_df)}")
    # print(f"GPT images shape: {gpt_images.shape}, {len(gpt_level_df)}")
    #
    # dataset = LevelDataset()
    #
    # split_ratio = 0.8
    # num_samples = len(human_level_df)
    # indices = np.arange(num_samples)
    # np.random.shuffle(indices)
    #
    # # calculate split index
    # split_point = int(split_ratio * num_samples)
    # seen_indices = indices[:split_point]
    # unseen_indices = indices[split_point:]
    #
    # human_seen_df = human_level_df.iloc[seen_indices].reset_index(drop=True)
    # human_unseen_df = human_level_df.iloc[unseen_indices].reset_index(drop=True)
    #
    # human_seen_images = human_images[seen_indices]
    # human_unseen_images = human_images[unseen_indices]
    #
    # human_seen_df['type'] = 'human_seen'
    # human_unseen_df['type'] = 'human_unseen'
    #
    # print(f"Human seen images shape: {human_seen_images.shape}, {len(human_seen_df)}")
    # dataset.add(human_seen_df, human_seen_images)
    # print(f"Human unseen images shape: {human_unseen_images.shape}, {len(human_unseen_df)}")
    # dataset.add(human_unseen_df, human_unseen_images)
    #
    # print(f"Dataset shape: {dataset.images.shape}, {len(dataset.dataframe)}")
    # dataset.add(rl_level_df, rl_images, train_ratio=0)
    # print(f"Dataset shape: {dataset.images.shape}, {len(dataset.dataframe)}")
    # dataset.add(gpt_level_df, gpt_images, train_ratio=0)
    # print(f"Dataset shape: {dataset.images.shape}, {len(dataset.dataframe)}")
    #
    # # save the daset to the pickle
    # with open("dataset.pkl", "wb") as f:
    #     import pickle
    #     pickle.dump(dataset, f)





    #

    # """
    # 2. Evaluate the Image Dataset with Evaluation Functions
    # """
    #
    # load the dataset from the pickle
    with open(join(dirname(__file__), "dataset.pkl"), "rb") as f:
        dataset = pickle.load(f)

    print(f"Preparing evaluators...")
    evaluations = {
        # 'cosine_res': partial(CosineEvaluator, backbone=resnet50(weights='DEFAULT')),
        # 'cosine_vgg': partial(CosineEvaluator, backbone=vgg11(weights='DEFAULT')),
        'cosine_vit': partial(CosineEvaluator, model_name='google/vit-base-patch16-224'),
        # 'ssim': partial(SSIMEvaluator, backbone=None),
    }

    task_filters = [1, 2, 3, 4, 5, None]
    type_filters = [('human_seen', 'human_seen'), ('human_seen', 'human_unseen'), ('human_seen', 'rl'),
                    ('human_seen', 'gpt'), ('rl', 'gpt')]

    results = list()

    for name, evaluator_cls in evaluations.items():
        for task_filter in task_filters:
            for type_filter in type_filters:
                comp1, comp2 = type_filter[0], type_filter[1]

                print(f"Running {name} evaluator, task: {task_filter} , {comp1} vs {comp2}...")

                image1 = dataset.set_filter(task_filter=task_filter, type_filter=comp1).get_images()
                image2 = dataset.set_filter(task_filter=task_filter, type_filter=comp2).get_images()

                print(f'Inputs: {image1.shape}, {image2.shape}')

                evaluator = evaluator_cls(gt_data=image1)
                score = evaluator.run(image2)

                results.append({
                    'evaluator': name,
                    'comparison1': comp1,
                    'comparison2': comp2,
                    'task': task_filter,
                    'type': type_filter,
                    'score': score
                })


    # concat the results to dataframe
    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv", index=False)

    print(tabulate(results_df, headers='keys', tablefmt='psql'))
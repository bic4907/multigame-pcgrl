import os
import pandas as pd
from glob import glob
import numpy as np
from chex import dataclass
from itertools import cycle
import json
import logging
from os.path import abspath, join, dirname, basename, splitext
from tqdm import tqdm
from functools import partial
from typing import List, Tuple
from transformers import CLIPProcessor
from PIL import Image
import random

import jax
import jax.numpy as jnp

from instruct_rl.utils.level_processing_utils import mutate_level_fn, map2onehot, add_coord_channel, add_coord_channel_batch
from instruct_rl.utils.img_preprocess import render_level_from_arr, clip_img_transition_preprocess
from instruct_rl.utils.sketch_preprocess import clip_sketch_batch_preprocess
from conf.config import EncoderConfig

log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))
logging.getLogger('absl').setLevel(logging.ERROR)


@dataclass
class CLIPDataset:
    class_ids: np.ndarray
    reward_cond: np.ndarray 
    input_ids: np.ndarray 
    attention_masks: np.ndarray
    pixel_values: np.ndarray
    sketch_values: np.ndarray
    augmentable: np.ndarray
    is_train: np.ndarray

@dataclass
class CLIPEmbedData:
    class_ids: np.ndarray
    state_embeddings: np.ndarray   
    text_embeddings: np.ndarray
    sketch_embeddings: np.ndarray
    
@dataclass
class CLIPContrastiveBatch:
    class_ids: np.ndarray
    input_ids: np.ndarray
    attention_mask: np.ndarray
    pixel_values: np.ndarray
    sketch_values: np.ndarray
    duplicate_matrix: np.ndarray  # (B, B) matrix indicating positive pairs
    

class CLIPDatasetBuilder:
    def __init__(self, config: EncoderConfig,
                 data_path: str,
                 instruct_csv: str,
                 processor:CLIPProcessor,
                 rng_key:jax.random.PRNGKey,
                 text_ratio:float,
                 state_ratio:float,
                 sketch_ratio:float,
                 train_shuffle:bool=False,
                 train_ratio:float=0.8,
                 max_len:int=77,
                 mutation_rate:float=0.1,
                 embed_type: str = "sub_condition",
                 aug_type: str = "bert"):
        self.config = config
        self.instruct_csv = instruct_csv
        self.data_path = data_path
        self.processor = processor
        self.max_len = max_len
        self.mutation_rate = mutation_rate
        self.embed_type = embed_type
        self.aug_type = aug_type
        
        self.train_ratio = train_ratio
        self.text_ratio = text_ratio
        self.state_ratio = state_ratio
        self.sketch_ratio = sketch_ratio
        self.train_shuffle = train_shuffle
        
        self.dataset = self._build_dataset(rng_key)
        
    def get_dataset(self) -> CLIPDataset:
        return self.dataset

    def _build_dataset(self, rng_key:jax.random.PRNGKey) -> CLIPDataset:
        instr_df, raw_instr_list, class_id2reward_cond, unique_reward_cond = self ._load_instr_csv()
        self.class_id2reward_cond = class_id2reward_cond
        
        logger.info(f"Unique reward conditions number: {len(unique_reward_cond)}")
                    
        dataset_dict = {
                "class_ids": [],
                "reward_cond": [],
                "input_ids": [],
                "attention_masks": [],
                "pixel_values": [],
                "sketch_values": [],
                "augmentable": [],
                "is_train": [],
            }

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"The directory {abspath(self.data_path)} does not exist. Check `data_path` in config.")

        logging.info(f"Loading goal imgs from {abspath(join(self.data_path))}")
        get_folder_name = lambda text: text.lower().replace('.', '').replace(' ', '_')
        
        for idx, instr_i in tqdm(enumerate(raw_instr_list), total=len(raw_instr_list)):
            folder_name_i = get_folder_name(instr_i)
            
            np_paths_raw = sorted(glob(join(self.data_path, "numpy", folder_name_i, "*.npy")))

            if len(np_paths_raw) == 0:
                logging.warning(f"No images found for instruction \"{instr_i}\".")
                continue

            sketch_dir = join(self.data_path, "sketch", folder_name_i)
            
            pairs = []
            for np_path in np_paths_raw:
                base = splitext(basename(np_path))[0]
                sketch_path = join(sketch_dir, base + ".png")
                if os.path.exists(sketch_path):
                    pairs.append({"state": np_path, "sketch": sketch_path})
                else:
                    logging.warning(f"Sketch file {sketch_path} does not exist for instruction \"{instr_i}\".")
            
            if not pairs:
                logging.warning(f"No matching state/sketch pairs for instruction \"{instr_i}\".")
                continue

            # Split pairs into train and validation before any processing
            rng_key, sub_key = jax.random.split(rng_key)
            
            shuffled_indices = jax.random.permutation(sub_key, len(pairs))
            
            num_train = int(len(pairs) * self.train_ratio)
            train_indices = shuffled_indices[:num_train]
            val_indices = shuffled_indices[num_train:]
            
            train_pairs = [pairs[i] for i in train_indices]
            val_pairs = [pairs[i] for i in val_indices]

            for is_train, pair_group in [(True, train_pairs), (False, val_pairs)]:
                if not pair_group:
                    continue
                
                state_ratio_apply = self.state_ratio if is_train else 1.0
                sketch_ratio_apply = self.sketch_ratio if is_train else 1.0

                num_states_to_use = max(1, int(len(pair_group) * state_ratio_apply ))
                num_sketches_to_use = max(1, int(len(pair_group) * sketch_ratio_apply))

                state_paths_subset = [p["state"] for p in pair_group[:num_states_to_use]]
                sketch_paths_subset = [p["sketch"] for p in pair_group[:num_sketches_to_use]]

                # Cycle subsets to match the original group size
                final_state_paths = [p for p, _ in zip(cycle(state_paths_subset), pair_group)]
                final_sketch_paths = [p for p, _ in zip(cycle(sketch_paths_subset), pair_group)]
                if is_train and self.train_shuffle:
                    # shuffle the final paths for training
                    random.shuffle(final_state_paths)
                n_instr_imgs = len(pair_group)

                if self.config.use_map_array:
                    map_array = np.array(self._load_map_array(final_state_paths, rng_key))
                else:
                    initial_imgs, goal_imgs = self._load_imgs(final_state_paths, rng_key)
                    preprocess_imgs = clip_img_transition_preprocess(initial_imgs, goal_imgs)
                    preprocess_imgs = np.array(preprocess_imgs)

                sketch_imgs = self._load_sketchs(final_sketch_paths, rng_key)
                preprocess_sketch = clip_sketch_batch_preprocess(sketch_imgs)
                preprocess_sketch = np.array(preprocess_sketch)
                preprocess_sketch = add_coord_channel_batch(preprocess_sketch)

                reward_cond_i = instr_df["reward_cond"].iloc[idx]
                class_id_i = instr_df["class_id"].iloc[idx]
                class_id_repeat = np.tile(class_id_i, (n_instr_imgs, 1))
                reward_cond_repeat = np.tile(reward_cond_i, (n_instr_imgs, 1))
                augmentable = np.where(reward_cond_i[0] == '5', 0, 1)
                augmentable_repeat = np.tile(augmentable, (n_instr_imgs,))
                is_train_repeat = np.tile(is_train, (n_instr_imgs,))
                
                text_subset_length = max(1, int(4*self.text_ratio))
                
                subset = instr_df[instr_df["reward_cond"].apply(lambda x: np.array_equal(x, reward_cond_i))]
                records = subset[["input_ids", "attention_masks"]][:text_subset_length].to_dict("records")
                records_cycle = cycle(records)
                
                instr_input_ids_repeat = np.array([next(records_cycle)["input_ids"] for _ in range(n_instr_imgs)])
                instr_attention_masks_repeat = np.array([next(records_cycle)["attention_masks"] for _ in range(n_instr_imgs)])
            
                # Data check
                data_elements = {
                    "class_ids": class_id_repeat,
                    "reward_cond": reward_cond_repeat,
                    "input_ids": instr_input_ids_repeat,
                    "attention_masks": instr_attention_masks_repeat,
                    "pixel_values": map_array if self.config.use_map_array else preprocess_imgs,
                    "sketch_values": preprocess_sketch,
                    "augmentable": augmentable_repeat,
                    "is_train": is_train_repeat
                }
                
                lengths = {name: len(data) for name, data in data_elements.items()}
                if len(set(lengths.values())) > 1:
                    logging.warning(f"Mismatch in data lengths for instruction '{instr_i}': {lengths}")
                    continue
                
                dataset_dict["class_ids"].extend(class_id_repeat)
                dataset_dict["reward_cond"].extend(reward_cond_repeat)
                dataset_dict["input_ids"].extend(instr_input_ids_repeat)
                dataset_dict["attention_masks"].extend(instr_attention_masks_repeat)
                if self.config.use_map_array:
                    dataset_dict["pixel_values"].extend(map_array)
                else:
                    dataset_dict["pixel_values"].extend(preprocess_imgs)
                dataset_dict["sketch_values"].extend(preprocess_sketch)
                dataset_dict["augmentable"].extend(augmentable_repeat)
                dataset_dict["is_train"].extend(is_train_repeat)

            #NOTE: for debug
            # if idx == 15:
            #     break
        logging.info(f"Finished loading dataset")
        return CLIPDataset(
                class_ids=np.array(dataset_dict["class_ids"]),
                reward_cond=np.array(dataset_dict["reward_cond"]),
                input_ids=np.array(dataset_dict["input_ids"]),
                attention_masks=np.array(dataset_dict["attention_masks"]),
                pixel_values=np.array(dataset_dict["pixel_values"]),
                sketch_values=np.array(dataset_dict["sketch_values"]),
                augmentable=np.array(dataset_dict["augmentable"]),
                is_train=np.array(dataset_dict["is_train"])
            )

    
    def _load_instr_csv(self) -> Tuple[List[str], np.ndarray, np.ndarray]:
        #Load instruction csv
        csv_path = abspath(join(dirname(__file__), '..', '..', 'instruct', f'{self.instruct_csv}.csv'))

        instr_df = pd.read_csv(csv_path)
        logging.info(f"Loading instruction csv from {csv_path}")
        
        ##get instruction
        language_instr_list = instr_df["instruction"].to_list()
        
        ##get reward_enum_list
        reward_enum_list = [[num] for num in instr_df["reward_enum"].to_list()]
        max_len = max(len(x) for x in reward_enum_list)
        reward_enum = np.array([
            x + [0] * (max_len - len(x)) for x in reward_enum_list
        ])

        ##get condition
        df_cond = instr_df.filter(regex='^condition_*')
        condition_df = df_cond.reindex(sorted(df_cond.columns, key=lambda x: int(x.split('_')[-1])), axis=1)  
        
        
        # get sub_condition
        df_sub_cond = instr_df.filter(regex='^sub_condition_*')
        sub_condition_df = df_sub_cond.reindex(sorted(df_sub_cond.columns, key=lambda x: int(x.split('_')[-1])), axis=1)     
         
        
        # generate reward_cond
        reward_cond = [[reward_i.tolist(), condition_df.iloc[i, reward_i-1].to_list(), sub_condition_df.iloc[i, reward_i-1].to_list()] for i, reward_i in enumerate(reward_enum)]
        reward_cond = np.array(reward_cond).squeeze()
        unique_reward_cond = np.unique(reward_cond, axis=0)
                    
        # Generate a mapping from unique class IDs to indices
        reward_cond2class_id= {tuple(unique_reward_cond[i].tolist()): i for i in range(len(unique_reward_cond))}
        class_id2reward_cond = {v: k for k, v in reward_cond2class_id.items()}
        class_id = [reward_cond2class_id[tuple(reward_cond[i])] for i in range(len(reward_cond))]
        class_id = np.array(class_id)
        
        tokenized_instrs = self.processor(
            text = language_instr_list,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=77
        )
        instr_input_ids, instr_attention_masks = tokenized_instrs['input_ids'], tokenized_instrs['attention_mask']
        instr_df = pd.DataFrame({
            "raw_instr": language_instr_list,
            "class_id": class_id,
            "reward_cond": [np.array(r) for r in reward_cond],
            "input_ids": [np.array(i) for i in instr_input_ids],
            "attention_masks": [np.array(a) for a in instr_attention_masks]
        })
        
        
        return instr_df, language_instr_list, class_id2reward_cond, unique_reward_cond
        
    def _load_imgs(self, np_paths: List[str], rng_key:jax.random.PRNGKey) -> List[jnp.ndarray]:
        initial_imgs = []
        goal_images = []
        img_renderer = partial(render_level_from_arr, tile_size=16, render_border=True)
        for np_path in np_paths:

            with open(np_path, "r", encoding="utf-8") as f:
                level_data = json.load(f)
            rng_key, sub_key = jax.random.split(rng_key)
            level_array = jnp.array(level_data)
            mutate_level = mutate_level_fn(level_array, rng_key=sub_key, mutation_rate=self.mutation_rate)

            rendered_initial_img = img_renderer(mutate_level)
            initial_imgs.append(rendered_initial_img)

            rendered_goal_img = img_renderer(level_array)
            goal_images.append(rendered_goal_img)

        return jnp.array(initial_imgs), jnp.array(goal_images)

    def _load_sketchs(self, sketch_paths: List[str], rng_key: jax.random.PRNGKey) -> List[jnp.ndarray]:
        sketch_images = []
        for sketch_path in sketch_paths:
            try:
                sketch_img = Image.open(sketch_path).convert("L")
                sketch_array = np.array(sketch_img)
                sketch_images.append(sketch_array)
            except Exception as e:
                print(f"Error loading image {sketch_path}: {e}")
        return jnp.array(sketch_images)

    def _load_map_array(self, np_paths: List[str], rng_key:jax.random.PRNGKey) -> jnp.ndarray:

        map_array_list = []
        for np_path in np_paths:

            level_data = np.load(np_path, allow_pickle=True)
            level_array = jnp.array(level_data)
            map_array = map2onehot(level_array)

            if map_array.ndim == 4:
                map_array = map_array[0]

            map_array = add_coord_channel(map_array)
            map_array_list.append(map_array)

        return jnp.array(map_array_list)
    

    def get_split_dataset(self):
        """
        Get train, test datasets based on the 'is_train' flag.
        Returns:
            Tuple[CLIPDataset, CLIPDataset]: Train and Test datasets.
        """
        train_mask = self.dataset.is_train
        test_mask = ~self.dataset.is_train

        train_dataset = CLIPDataset(
            class_ids=self.dataset.class_ids[train_mask],
            reward_cond=self.dataset.reward_cond[train_mask],
            input_ids=self.dataset.input_ids[train_mask],
            attention_masks=self.dataset.attention_masks[train_mask],
            pixel_values=self.dataset.pixel_values[train_mask],
            sketch_values=self.dataset.sketch_values[train_mask],
            augmentable=self.dataset.augmentable[train_mask],
            is_train=self.dataset.is_train[train_mask]
        )
        test_dataset = CLIPDataset(
            class_ids=self.dataset.class_ids[test_mask],
            reward_cond=self.dataset.reward_cond[test_mask],
            input_ids=self.dataset.input_ids[test_mask],
            attention_masks=self.dataset.attention_masks[test_mask],
            pixel_values=self.dataset.pixel_values[test_mask],
            sketch_values=self.dataset.sketch_values[test_mask],
            augmentable=self.dataset.augmentable[test_mask],
            is_train=self.dataset.is_train[test_mask]
        )

        return train_dataset, test_dataset   
   
    def get_class_id2reward_cond(self):
        """
        Get the mapping of class IDs to reward conditions.
        Returns:
            dict: Mapping of class IDs to reward conditions.
        """
        return self.class_id2reward_cond
 
def create_clip_batch(dataset: CLIPDataset, batch_size: int, rng_key: jax.random.PRNGKey, augment: bool = False) -> CLIPContrastiveBatch:
    """
    Create a batch of CLIP data for contrastive learning.
    Args:
        dataset (CLIPDataset): The dataset to sample from.
        batch_size (int): The size of the batch to create.
        rng_key (jax.random.PRNGKey): Random key for sampling.
        augment (bool): Whether to apply augmentations to the pixel_values. Default is False.
    Returns:
        CLIPContrastiveBatch: A batch of CLIP data.
    """
    
    n_samples = len(dataset.input_ids)
    shuffled_indices = jax.random.permutation(rng_key, n_samples)
    shuffled_indices = np.array(shuffled_indices)
    
    for start_idx in range(0,n_samples,batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = shuffled_indices[start_idx:end_idx]
        if len(batch_indices) < batch_size:
            needed = batch_size - len(batch_indices)
            extra_indices = np.random.choice(n_samples, needed, replace=True)
            batch_indices = np.concatenate([batch_indices, extra_indices])
            
        
        class_ids = dataset.class_ids[batch_indices].squeeze()             #(B,)
        input_ids = dataset.input_ids[batch_indices]           #(B,T) -> (B,77)
        attention_mask = dataset.attention_masks[batch_indices] #(B,T) -> (B,77)
        pixel_values = dataset.pixel_values[batch_indices]     #(B,H,W,C) -> (B,16,16,5)
        sketch_values = dataset.sketch_values[batch_indices]   #(B,H,W,C) -> (B,224,224,3)
        duplicate_matrix = np.equal.outer(class_ids, class_ids).astype(np.float32) # (B, B)
        augmentable = dataset.augmentable[batch_indices]

        if augment:
            augment_mask = augmentable.astype(bool)
            k = np.random.choice([0, 1, 2, 3])  # Number of 90-degree rotations

            pixel_values[augment_mask] = np.rot90(pixel_values[augment_mask], k=k, axes=(1, 2))
            sketch_values[augment_mask] = np.rot90(sketch_values[augment_mask], k=k, axes=(1, 2))

            if np.random.rand() > 0.5:
                pixel_values[augment_mask] = np.flip(pixel_values[augment_mask], axis=1)
                sketch_values[augment_mask] = np.flip(sketch_values[augment_mask], axis=1)

            if np.random.rand() > 0.5:
                pixel_values[augment_mask] = np.flip(pixel_values[augment_mask], axis=2)
                sketch_values[augment_mask] = np.flip(sketch_values[augment_mask], axis=2)

        yield CLIPContrastiveBatch(
            class_ids=class_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            sketch_values=sketch_values,
            duplicate_matrix=duplicate_matrix
        )

def create_clip_embedding_table(embed_queue, reward_df: pd.DataFrame):
    pass
# def create_clip_embedding_table(embed_queue, reward_df: pd.DataFrame) -> wandb.Table:
#     reward_ids = [e.reward_id for e in embed_queue]
#     embeds = np.array([e.embedding for e in embed_queue])

#     # TSNE (2dim)
#     tsne = TSNE(n_components=2, random_state=42)
#     tsne_embeds = tsne.fit_transform(embeds)

#     inst_cols = reward_df.iloc[reward_ids][['instruction', 'reward_enum']].reset_index()
#     tsne_df = pd.DataFrame(tsne_embeds, columns=['tsne_x', 'tsne_y']).reset_index()
#     df = pd.concat([inst_cols, tsne_df], axis=1).drop(columns=['index'])

#     # WandB table, logging
#     train_table = wandb.Table(dataframe=df) 
    
if __name__ == "__main__":
    data_path = "/app/nas/huamn_ipcgrl_dataset"
    instruct_csv = "sub_condition/bert/scn-1_se-whole"
    rng_key = jax.random.PRNGKey(0)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    config = EncoderConfig()
    dataset_builder = CLIPDatasetBuilder(config, data_path, instruct_csv, processor, rng_key=rng_key, train_ratio=0.8, sketch_ratio=1.0, state_ratio=1.0, max_len=77, mutation_rate=0.1, embed_type="sub_condition", aug_type="bert")
    
    train_dataset, test_dataset = dataset_builder.get_split_dataset()
    
    print(f"Total samples: {len(dataset_builder.get_dataset().class_ids)}")
    print(f"Train samples: {len(train_dataset.class_ids)}")
    print(f"Test samples: {len(test_dataset.class_ids)}")
    
    for batch in create_clip_batch(train_dataset, batch_size=32, rng_key=rng_key):
        print("\n--- Train Batch ---")
        print("Input IDs shape:", batch.input_ids.shape)
        print("Attention Mask shape:", batch.attention_mask.shape)
        print("Pixel Values shape:", batch.pixel_values.shape)
        print("Sketch Values shape:", batch.sketch_values.shape)
        print("Duplicate Matrix shape:", batch.duplicate_matrix.shape)
        rng_key, subkey = jax.random.split(rng_key)
        break
    
    for batch in create_clip_batch(test_dataset, batch_size=32, rng_key=rng_key):
        print("\n--- Test Batch ---")
        print("Input IDs shape:", batch.input_ids.shape)
        print("Attention Mask shape:", batch.attention_mask.shape)
        print("Pixel Values shape:", batch.pixel_values.shape)
        print("Sketch Values shape:", batch.sketch_values.shape)
        print("Duplicate Matrix shape:", batch.duplicate_matrix.shape)
        rng_key, subkey = jax.random.split(rng_key)
        break
    
    
    
    
    
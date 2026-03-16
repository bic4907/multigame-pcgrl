import numpy as np
import pandas as pd
from os.path import join, dirname, abspath
from glob import glob
from tqdm import tqdm
import jax.numpy as jnp
from instruct_rl.evaluate import get_loss_batch

# Load the numpy files from the vision_level_gpt directory
level_dir = join(dirname(abspath(__file__)), "vision_level_gpt", "numpy")
level_files = glob(join(level_dir, "**", "*.npy"), recursive=True)
print(f"Found {len(level_files)} level files. ({level_dir})")

# Load the instruction file
instruction_file = abspath(join(dirname(abspath(__file__)), "..", "..", "instruct", "test", "bert", "scn-1_se-whole.csv"))
inst_df = pd.read_csv(instruction_file)
inst_df['instruct_dir'] = inst_df['instruction'].apply(lambda x: x.replace(' ', '_').replace('.', '').lower())

print(f"Loaded {len(inst_df)} instructions from {instruction_file}.")
# remove the columns starts with embed_*
embed_cols = [col for col in inst_df.columns if col.startswith('embed_')]
inst_df = inst_df.drop(columns=embed_cols )
# remove the columns starts with condition_5 to condition_8
removal_cond_cols = [col for col in inst_df.columns if col.startswith('condition_5') or col.startswith('condition_6')
                     or col.startswith('condition_7') or col.startswith('condition_8')]
inst_df = inst_df.drop(columns=removal_cond_cols)
inst_df = inst_df.drop(columns=['train', 'instruction'])

level_data = {
    'instructions': [],
    'reward_i': [],
    'condition': [],
    'conditions': [],
    'env_maps': [],
    'level_files': [],
}

# Process the level files
for level_file in tqdm(level_files):
    level_arr = np.load(level_file)

    level_file_arr = level_file.split("/")
    level_instruct = level_file_arr[-2]

    # find the row from the inst_df that matches the level instruction
    level_row = inst_df[inst_df['instruct_dir'] == level_instruct]

    level_enum = level_row['reward_enum'].values[0]

    # get the conditions ['condition_0', 'condition_1', 'condition_2', 'condition_3', 'condition_4']
    level_cond = level_row.filter(regex='condition_').values[0].tolist()
    level_cond_uni = level_row[f'condition_{level_enum - 1}'].values[0]

    level_data['instructions'].append(level_instruct)
    level_data['reward_i'].append(level_enum)
    level_data['conditions'].append(level_cond)
    level_data['condition'].append(level_cond_uni)
    level_data['env_maps'].append(level_arr)
    level_data['level_files'].append(level_file)

# change the jnp.array
level_data_jnp = {
    'reward_i': jnp.array(level_data['reward_i']).reshape(-1, 1),  # Ensure it's a 2D array
    'conditions': jnp.array(level_data['conditions']).reshape(-1, 5),
    'env_maps': jnp.array(level_data['env_maps']),
}
print(f"reward_i shape: {level_data_jnp['reward_i'].shape}, conditions shape: {level_data_jnp['conditions'].shape}, "
      f"env_maps shape: {level_data_jnp['env_maps'].shape}")

opposite_conditions = (level_data_jnp['conditions'] + 2) % 4

values = get_loss_batch(reward_i=level_data_jnp['reward_i'],
                         condition=level_data_jnp['conditions'],
                         env_maps=level_data_jnp['env_maps']).value
opposite_values = get_loss_batch(reward_i=level_data_jnp['reward_i'],
                                    condition=opposite_conditions,
                                    env_maps=level_data_jnp['env_maps']).value

level_data['actual'] = values.reshape(-1).tolist()  # Flatten the values to match the number of levels
level_data['actual_opposite'] = opposite_values.reshape(-1).tolist()  # Flatten the opposite values
# Create a DataFrame to store the results
result_df = pd.DataFrame(level_data)

# if the reward_i is 5, then set actual = actual - actual_opposite
result_df['actual'] = result_df.apply(
    lambda row: row['actual'] - row['actual_opposite'] if row['reward_i'] == 5 else row['actual'], axis=1)
# remove the actual_opposite column
result_df = result_df.drop(columns=['actual_opposite'])

# remove the conditions and env_maps columns
result_df = result_df.drop(columns=['conditions', 'env_maps'])
# Save the DataFrame to a CSV file
result_csv = join("human_level_fitness.csv")
result_df.to_csv(result_csv, index=False)

# group-by statistics
# sort the reward_i and condition columns
result_df = result_df.sort_values(by=['reward_i', 'condition'])
grouped_df = result_df.groupby(['reward_i', 'condition']).agg({'actual': ['mean', 'std', 'count']}).reset_index()
grouped_df.columns = ['reward_i', 'condition', 'mean', 'std', 'count']

# Save the grouped DataFrame to a CSV file
grouped_csv = join("human_level_fitness_grouped.csv")
grouped_df.to_csv(grouped_csv, index=False)
print(f"Results saved to {result_csv} and {grouped_csv}")





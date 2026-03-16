import pandas as pd
from os.path import join, dirname, abspath
import re


INITIAL_STATE = abspath(join(dirname(abspath(__file__)), '..', '..',  'notebook', 'results', 'initial_state.csv'))

def get_initial_state() -> pd.DataFrame:
    initial_state = pd.read_csv(INITIAL_STATE)

    # columns start with feat_
    index_cols = ['seed']
    feat_cols = [col for col in initial_state.columns if col.startswith('feat_')]
    cols = index_cols + feat_cols

    df = initial_state[cols]
    # change the column name feat_ to s0_feat_
    cols = [col.replace('feat_', 's0_feat_') for col in cols]
    df.columns = cols

    return df

def calculate_progress(row: pd.Series, condition_col: str, feat_col: str) -> float:
    # get the current state

    if row[condition_col] == -1:
        return -1.0

    # print(row[feat_col], row[condition_col], row[f's0_{feat_col}'], row[condition_col])

    progress = 1 - abs((row[condition_col] - row[feat_col]) / (row[condition_col] - row[f's0_{feat_col}'] + 0.0000001))

    progress = progress * 100
    progress = max(0, min(100, progress))

    # print(f'row[feat_col]: {row[feat_col]}, row[condition_col]: {row[condition_col]}, row[s0_{feat_col}]: {row[f"s0_{feat_col}"]}, row[condition_col]: {row[condition_col]}, progress: {progress}')
    # print(f'row[feat_col]: {row[feat_col]}, row[s0_{feat_col}]: {row[f"s0_{feat_col}"]} /  row[condition_col]: {row[condition_col]}, row[s0_{feat_col}]: {row[f"s0_{feat_col}"]} = progress: {progress}')

    return progress

def calculate_direction(row: pd.Series, condition_col: str) -> float:
    cond = row[condition_col]

    if row[condition_col] == -1:
        return -1.0

    num = row[f"feat_batdir{int(cond)}"]
    oppo = row[f"feat_batdir{int((cond + 2) % 4)}"]
    total = num + oppo

    return num / total * 100 if total != 0 else 0

def process_result(raw_df: pd.DataFrame, div_df: pd.DataFrame,
                   return_ablation_columns: bool = False,
                   groupby_trainset: bool = False) -> pd.DataFrame:

    df = raw_df.copy()

    initial_df = get_initial_state()
    df = pd.merge(df, initial_df, on='seed')

    optional_columns = {
        'human_likeness': 0.1,
    }

    for col, default_val in optional_columns.items():
        if col not in df.columns:
            df[col] = default_val
    # ----        
            
            
    # get progress for each feature
    df['progress_region'] = df.apply(lambda row: calculate_progress(row, 'condition_0', 'feat_region'), axis=1)
    df['progress_plength'] = df.apply(lambda row: calculate_progress(row, 'condition_1', 'feat_plength'), axis=1)
    df['progress_nblock'] = df.apply(lambda row: calculate_progress(row, 'condition_2', 'feat_nblock'), axis=1)
    df['progress_nbat'] = df.apply(lambda row: calculate_progress(row, 'condition_3', 'feat_nbat'), axis=1)
    df['progress_batdir'] = df.apply(lambda row: calculate_direction(row, 'condition_4'), axis=1)


    df['buffer_ratio'] = df['config.buffer_ratio']
    df['encoder_size'] = df['config.encoder.output_dim']
    df['exp_id'] = df['config.run_id']
    df['exp_seed'] = df['config.seed']
    df['instruct'] = df['config.instruct']
    df['trainset'] = df['train']
    df['eval_modality'] = df['config.eval_modality']
    df['sim_coef'] = df['config.SIM_COEF']

    # remove columns starts with config
    config_cols = [col for col in df.columns if col.startswith('config')]
    # remove index columns
    index_cols = ['index', 'loss', 'index.1', 'condition_5', 'condition_6', 'condition_7', 'condition_8']
    config_cols = config_cols + index_cols
    # remove anonymous columns and index
    change_dict = {'condition_0': 'cond_region', 'condition_1': 'cond_plength', 'condition_2': 'cond_nblock',
                   'condition_3': 'cond_nbat', 'condition_4': 'cond_batdir'}
    df = df.rename(columns=change_dict)

    df = df.drop(columns=config_cols).reset_index(drop=True)

    if return_ablation_columns:
        ablation_col = ['buffer_ratio', 'encoder_size']
    else:
        ablation_col = list()

    groupby_col = list()
    if groupby_trainset is True:
        groupby_col = ['trainset']

    # change the column name
    df = df.groupby(['exp_id', 'exp_seed', 'instruct', 'row_i', 'reward_enum', 'eval_modality', 'sim_coef',
                     'cond_region', 'cond_plength', 'cond_nblock', 'cond_nbat', 'cond_batdir'
                     ] + ablation_col + groupby_col).agg({'progress_region': 'mean',
                                                   'progress_plength': 'mean',
                                                   'progress_nblock': 'mean',
                                                   'progress_nbat': 'mean',
                                                   'progress_batdir': 'mean',
                                                          'human_likeness': 'mean', 'tpkldiv': 'mean'
                                                          }).reset_index()

    div_df = div_df[['config.run_id', 'row_i', 'diversity']]
    div_df.columns = ['exp_id', 'row_i', 'diversity']
    df = pd.merge(df, div_df, on=['exp_id', 'row_i'])

    return df

def process_result_from_loss_csv(raw_df: pd.DataFrame, exp_name:str,
                   return_ablation_columns: bool = False,
                   groupby_trainset: bool = False) -> pd.DataFrame:

    df = raw_df.copy()

    initial_df = get_initial_state()
    df = pd.merge(df, initial_df, on='seed')
    

    optional_columns = {
        'human_likeness': 0.1,
    }

    for col, default_val in optional_columns.items():
        if col not in df.columns:
            df[col] = default_val
            
            
    # get progress for each feature
    df['progress_region'] = df.apply(lambda row: calculate_progress(row, 'condition_0', 'feat_region'), axis=1)
    df['progress_plength'] = df.apply(lambda row: calculate_progress(row, 'condition_1', 'feat_plength'), axis=1)
    df['progress_nblock'] = df.apply(lambda row: calculate_progress(row, 'condition_2', 'feat_nblock'), axis=1)
    df['progress_nbat'] = df.apply(lambda row: calculate_progress(row, 'condition_3', 'feat_nbat'), axis=1)
    df['progress_batdir'] = df.apply(lambda row: calculate_direction(row, 'condition_4'), axis=1)


    seed_match = re.search(r"s-(\d+)", exp_name)
    exp_seed = int(seed_match.group(1)) if seed_match else None
    # instruct: scn-1_se-1
    instruct_match = re.search(r"scn-1_se-\d+", exp_name)
    instruct = instruct_match.group(0) if instruct_match else None

    sim_match = re.search(r"cf-([\d.]+)", exp_name)
    sim_coef = float(sim_match.group(1)) if sim_match else None


    df['buffer_ratio'] = 1.0
    df['encoder_size'] = 64
    df['exp_id'] = exp_name
    df['exp_seed'] = exp_seed
    df['instruct'] = instruct
    df['trainset'] = df['train']
    df['eval_modality'] = 'text'
    df['sim_coef'] =  sim_coef

    # remove columns starts with config
    config_cols = [col for col in df.columns if col.startswith('config')]
    # remove index columns
    index_cols = ['index', 'loss', 'index.1', 'condition_5', 'condition_6', 'condition_7', 'condition_8']
    config_cols = config_cols + index_cols
    # remove anonymous columns and index
    change_dict = {'condition_0': 'cond_region', 'condition_1': 'cond_plength', 'condition_2': 'cond_nblock',
                   'condition_3': 'cond_nbat', 'condition_4': 'cond_batdir'}
    df = df.rename(columns=change_dict)

    df = df.drop(columns=config_cols).reset_index(drop=True)

    if return_ablation_columns:
        ablation_col = ['buffer_ratio', 'encoder_size']
    else:
        ablation_col = list()

    groupby_col = list()
    if groupby_trainset is True:
        groupby_col = ['trainset']

    # change the column name
    df = df.groupby(['exp_id', 'exp_seed', 'instruct', 'row_i', 'reward_enum', 'eval_modality', 'sim_coef',
                     'cond_region', 'cond_plength', 'cond_nblock', 'cond_nbat', 'cond_batdir'
                     ] + ablation_col + groupby_col).agg({'progress_region': 'mean',
                                                   'progress_plength': 'mean',
                                                   'progress_nblock': 'mean',
                                                   'progress_nbat': 'mean',
                                                   'progress_batdir': 'mean',
                                                          'human_likeness': 'mean', 'tpkldiv': 'mean'
                                                          }).reset_index()


    return df

AGG_VALUES = {'progress_region': 'mean', 'progress_plength': 'mean', 'progress_nblock': 'mean', 'progress_nbat': 'mean', 'progress_batdir': 'mean'}

PROGRESS_COL_DICT = {
    1: 'progress_region',
    2: 'progress_plength',
    3: 'progress_nblock',
    4: 'progress_nbat',
    5: 'progress_batdir'
}
COND_COL_DICT = {
    1: 'cond_region',
    2: 'cond_plength',
    3: 'cond_nblock',
    4: 'cond_nbat',
    5: 'cond_batdir'
}
COND_NAME_DICT = {
    1: 'Region',
    2: 'Path Length',
    3: 'Wall Count',
    4: 'Bat Count',
    5: 'Bat Direction'
}


def get_progress_result(df: pd.DataFrame, instruct: str, reward_enum: int, agg_values: dict = AGG_VALUES,
                        return_cond_col=True, return_seed_col=False, return_renamed_col: bool = True,
                        return_diversity=False, return_ablation_col=False) -> pd.DataFrame:
    _df = df[(df['instruct'] == instruct) | (df['method'] == 'random')]
    _df = _df[_df['reward_enum'] == reward_enum]

    progress_col_dict = {
        1: 'progress_region',
        2: 'progress_plength',
        3: 'progress_nblock',
        4: 'progress_nbat',
        5: 'progress_batdir'
    }
    cond_col_dict = {
        1: 'cond_region',
        2: 'cond_plength',
        3: 'cond_nblock',
        4: 'cond_nbat',
        5: 'cond_batdir'
    }

    progress_col = progress_col_dict[reward_enum]
    cond_col = cond_col_dict[reward_enum]

    se1_df = _df[_df['reward_enum'] == reward_enum]

    ablation_col = list()
    if return_ablation_col:
        ablation_col = ['buffer_ratio', 'encoder_size']

    gropuby_cols = ['method', 'instruct'] + (['exp_seed'] if return_seed_col else []) + (
        [cond_col] if return_cond_col else []) + (ablation_col if return_ablation_col else [])

    diversity_col = list()
    if return_diversity:
        agg_values['diversity'] = 'mean'
        diversity_col = ['diversity']

    grouped = se1_df.groupby(gropuby_cols).agg(agg_values).reset_index()
    grouped = grouped[[*gropuby_cols, progress_col, *diversity_col]]

    # rename the cond col and progress col to rename
    if return_renamed_col:
        grouped.rename(columns={progress_col: 'progress', cond_col: 'cond'}, inplace=True)

    grouped.insert(2, 'task', cond_col)

    # order the datataframe to 'random', 'cont', 'rawobs', 'embed', 'instruct' order
    grouped['method'] = pd.Categorical(grouped['method'], ['random',
                                                           'cont', 'rawobs', 'embed', 'instruct',
                                                           'cont_2c', 'rawobs_2c', 'embed_2c', 'instruct_2c'])
    grouped.sort_values('method', inplace=True)
    grouped.reset_index(drop=True, inplace=True)

    return grouped



def get_progress_result_multi(df: pd.DataFrame, instruct: str, reward_enum: int,
                              return_diversity=True, return_cond_col=True) -> pd.DataFrame:
    _df = df[(df['instruct'] == instruct) | (df['method'] == 'random')]
    _df = _df[_df['reward_enum'] == reward_enum]

    progress_col_dict = {
        1: 'progress_region',
        2: 'progress_plength',
        3: 'progress_nblock',
        4: 'progress_nbat',
        5: 'progress_batdir'
    }
    cond_col_dict = {
        1: 'cond_region',
        2: 'cond_plength',
        3: 'cond_nblock',
        4: 'cond_nbat',
        5: 'cond_batdir'
    }

    reward_digits = [int(digit) for digit in str(reward_enum)]

    for i, reward in enumerate(reward_digits, start=1):
        if reward in progress_col_dict:
            progress_col = progress_col_dict[reward]
            cond_col = cond_col_dict[reward]
        else:
            continue

        _df[f'cond_{i}'] = _df[cond_col]
        _df[f'progress_{i}'] = _df[progress_col]

        _df.insert(2, f'task_{i}', cond_col)

    # _df.drop(columns=list(COND_COL_DICT.values()) + list(PROGRESS_COL_DICT.values()), inplace=True)

    gropuby_cols = ['method', 'instruct'] + [f'task_{i + 1}' for i in range(len(str(reward_enum)))]
    if return_cond_col:
        gropuby_cols += [f'cond_{i + 1}' for i in range(len(str(reward_enum)))]

    agg_values = {'progress_1': 'mean', 'progress_2': 'mean'}

    if return_diversity:
        agg_values['diversity'] = 'mean'

    grouped = _df.groupby(gropuby_cols).agg(agg_values).reset_index()

    grouped['method'] = pd.Categorical(grouped['method'], ['random',
                                                           'cont', 'rawobs', 'embed', 'instruct',
                                                           'cont_2c', 'rawobs_2c', 'embed_2c', 'instruct_2c'])
    grouped.sort_values('method', inplace=True)
    grouped.reset_index(drop=True, inplace=True)

    grouped['progress'] = (grouped['progress_1'] + grouped['progress_2']) / 2

    return grouped


if __name__ == '__main__':

    # print all dataframe columns
    pd.set_option('display.max_columns', None)

    RESULT_DIR = 'results'

    run_id = 'embed-bert_inst-scn-1_se-14_model-contconv_exp-def_vec_s-0--embed-bert_inst-whole-20250302144822'

    raw_path = join(abspath('..'), '..', 'notebook', RESULT_DIR, 'eval_comb_cont_pcgrl', run_id, 'raw.csv')
    raw_df = pd.read_csv(raw_path)

    div_path = join(abspath('..'), '..', 'notebook', RESULT_DIR, 'eval_comb_cont_pcgrl', run_id, 'diversity.csv')
    div_df = pd.read_csv(div_path)

    raw_df = process_result(raw_df, div_df)
    print(raw_df.head())

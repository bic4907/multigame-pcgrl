import wandb
from conf.config import Config

def get_wandb_name(config: Config):
    exp_dir_path = config.exp_dir
    # split by directory
    exp_dirs = exp_dir_path.split('/')

    return exp_dirs[-1]

def get_image_url(entity: str, project: str, run, reward_i, seed):
    key = f'Image/reward_{reward_i}/seed_{seed}'
    history = run.history(keys=[key])
    image_path = history[key][0]['path']

    image_url = f"https://api.wandb.ai/files/{entity}/{project}/{run.id}/{image_path}"
    return image_url


def get_run_by_id(entity: str, project: str, run_id: str):
    api = wandb.Api(timeout=600)
    return api.run(f"{entity}/{project}/{run_id}")


def start_wandb(config: Config):
    from instruct_rl.utils.env_loader import get_wandb_key

    wandb_key = get_wandb_key() or wandb.api.api_key

    if wandb_key and config.wandb_project:

        wandb.login(key=wandb_key)
        run = wandb.init(
            project=config.wandb_project,
            resume=config.wandb_resume,
            id=get_wandb_name(config),
            name=get_wandb_name(config),
            save_code=True)

        wandb.define_metric("Evaluation/llm_iteration")
        # define which metrics will be plotted against it
        wandb.define_metric("Evaluation/*", step_metric="Evaluation/llm_iteration")

        wandb.define_metric("train/step")
        wandb.define_metric("Iteration*", step_metric="train/step")

        wandb.config.update(dict(config), allow_val_change=True)


def finish_wandb():
    if wandb.run is not None:
        wandb.finish()
import os
import wandb
import pandas as pd
from tensorboardX import SummaryWriter

from instruct_rl.utils.logger import get_wandb_name

# Base logging handler
class BaseLoggingHandler:
    def __init__(self, **kwargs):
        self.train_start_time = None
        self.steps_prev_complete = 0
        self.config = kwargs.get("config", None)
        self.logger = kwargs.get("logger", None)

    def initialize(self):
        pass

    def set_start_time(self, start_time):
        """Sets the start time for training."""
        self.train_start_time = start_time

    def set_steps_prev_complete(self, steps):
        """Sets the previous steps completed."""
        self.steps_prev_complete = steps

    def log(self, metric, t):
        """Logs the metric."""
        raise NotImplementedError


# TensorBoard logging handler
class TensorBoardLoggingHandler(BaseLoggingHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer = SummaryWriter(self.config.exp_dir)

    def log(self, metric, t):
        for key, val in metric.items():
            if isinstance(val, (int, float)):  # Log scalars
                self.writer.add_scalar(key, val, t)
            elif isinstance(val, str):  # Log text
                self.writer.add_text(key, val, t)
            elif isinstance(val, list) and isinstance(val[0],
                                                      (list, str, int, float)):  # Handle images or lists of images
                self.writer.add_image(key, val, t)

    def add_text(self, category, text, t=0):
        self.writer.add_text(category, text, t)


# wandb logging handler
class WandbLoggingHandler(BaseLoggingHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if wandb.run is None:
            if self.config.wandb_key and self.config.wandb_project:

                wandb.login(key=self.config.wandb_key)
                wandb.init(project=self.config.wandb_project, name=get_wandb_name(self.config), save_code=True)
                wandb.config.update(dict(self.config), allow_val_change=True)

                if self.logger is not None:
                    self.logger.info(f"Initialized wandb with project {self.config.wandb_project}")
                else:
                    print(f"Initialized wandb with project {self.config.wandb_project}")

                self.use_wandb = True
            else:
                if self.logger is not None:
                    self.logger.info("wandb not initialized")
                else:
                    print("wandb not initialized")
                self.use_wandb = False
        else:
            self.use_wandb = True
            # logger is defined, wandb is already initialized
            if self.logger is not None:
                self.logger.info("Using existing wandb run: {}".format(wandb.run.id))

    def log(self, metric, t):
        if not self.use_wandb:
            return

        wandb.log(metric)

    def add_text(self, category, text, t=0):
        if not self.use_wandb:
            return

        def text_to_html(text):
            html_text = text.replace('\n', '<br>')

            html_text = html_text.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')

            return html_text

        html_output = text_to_html(text)
        wandb.log({category: wandb.Html(html_output)})


# CSV logging handler
class CSVLoggingHandler(BaseLoggingHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def log(self, metric, t):
        # Flatten all the data into strings (text or scalar) for CSV compatibility
        scalar_metric = {k: str(v) if not isinstance(v, (int, float)) else v for k, v in metric.items()}

        # Load or create a CSV dataframe
        csv_path = os.path.join(self.config.exp_dir, "progress.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame()

        scalar_metric['timestep'] = t

        # Append new data
        new_data = pd.DataFrame([scalar_metric])
        df = pd.concat([df, new_data], ignore_index=True)

        # Save the updated CSV
        df.to_csv(csv_path, index=False)


# Multiple logging handler
class MultipleLoggingHandler:
    def __init__(self, handler_classes: list, **kwargs):
        self.handlers = [handler_class(**kwargs) for handler_class in handler_classes]

    def set_start_time(self, start_time):
        for handler in self.handlers:
            handler.set_start_time(start_time)

    def set_steps_prev_complete(self, steps):
        for handler in self.handlers:
            handler.set_steps_prev_complete(steps)

    def log(self, metric, t):
        for handler in self.handlers:
            handler.log(metric, t)

    def add_text(self, text, t):
        for handler in self.handlers:
            if hasattr(handler, "add_text"):
                handler.add_text(text, t)

import numpy as np
import cv2
from tqdm import tqdm

from instruct_rl.vision.data.render import render_level, render_array_batch

human_levels = np.load("0418.npz", allow_pickle=True)['levels']

human_levels = human_levels[15, ...]



for level in tqdm(human_levels):
    # add axis in the first dimension
    level = np.expand_dims(level, axis=0)

    rendered_img = render_array_batch(level, tile_size=16)
    rendered_img = cv2.cvtColor(rendered_img[0], cv2.COLOR_RGB2BGR)
    cv2.imshow("Level", rendered_img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
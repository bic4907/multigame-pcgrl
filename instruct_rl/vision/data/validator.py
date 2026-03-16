import argparse
from glob import glob

import pygame
import jax.numpy as jnp
import multiprocessing as mp

from instruct_rl.vision.data.utils.pygame import *

# setting tile size/color
tile_size = 32
colors = {1: (200, 200, 200), 2: (50, 50, 50), 3: (200, 50, 50)}
tile_types = [1, 2, 3]

pygame.init()

class LossComputeProcess(mp.Process):
    def __init__(self, task_queue, result_queue):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        while True:
            level = self.task_queue.get()
            if level is None:
                break
            try:
                reward_i = jnp.array([[1]])
                condition = np.zeros((1, 5), dtype=int)
                level_batch = level[np.newaxis, :, :]
                results = get_loss_batch(reward_i=reward_i, condition=condition, env_maps=level_batch)
                self.result_queue.put(results.feature[0])
            except Exception as e:
                print("Loss compute error:", e)
                self.result_queue.put(None)

def main(config, inst_df = None, inst_filter = None):
    print("Loading files at ", config.dir)
    files = glob(os.path.join(config.dir, "numpy", "**", "*.npy"), recursive=True)
    print(f"Found {len(files)} files.")

    if not files:
        print("No numpy files found.")
        return

    if inst_filter is not None:
        inst_filter = [i.replace(" ", "_").replace(".", "").lower() for i in inst_filter]
        files = [f for f in files if any(i in f for i in inst_filter)]

    screen = pygame.display.set_mode((16 * tile_size, 16 * tile_size))

    if not files:
        print("No numpy files found.")
        return

    running = True
    current_index = 0
    current_tile = 1
    mouse_held = False
    status = []
    level_dirty = False

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    loss_proc = LossComputeProcess(task_queue, result_queue)
    loss_proc.start()

    level, raw_level, instruction, path, cond = load_level_from_arr(files, current_index, inst_df)
    task_queue.put(level.copy())
    waiting_for_result = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    current_tile = 1
                elif event.key == pygame.K_2:
                    current_tile = 2
                elif event.key == pygame.K_3:
                    current_tile = 3
                elif event.key == pygame.K_5:
                    current_index = int(input('Enter index: ')) - 1
                    current_index = min(max(current_index, 0), len(files) - 1)
                    level, raw_level, instruction, path, cond = load_level_from_arr(files, current_index, inst_df)
                    level_dirty = True
                    status = []
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_TAB:
                    current_index = (current_index + 1) % len(files)
                    level, raw_level, instruction, path, cond = load_level_from_arr(files, current_index, inst_df)
                    level_dirty = True
                    status = []
                elif event.key == pygame.K_LEFT:
                    current_index = (current_index - 1) % len(files)
                    level, raw_level, instruction, path, cond = load_level_from_arr(files, current_index, inst_df)
                    level_dirty = True
                    status = []
                elif event.key == pygame.K_r:
                    level, raw_level, instruction, path, cond = load_level_from_arr(files, current_index, inst_df)
                    level_dirty = True
                elif event.key == pygame.K_d:
                    files = delete_level(files, path)
                    current_index = (current_index - 1) % len(files)
                    level, raw_level, instruction, path, cond = load_level_from_arr(files, current_index, inst_df)
                    level_dirty = True
                elif event.key == pygame.K_s:
                    save_level(level, path)
                    level, raw_level, instruction, path, cond = load_level_from_arr(files, current_index, inst_df)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_held = True
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_held = False
                level_dirty = True

        if mouse_held:
            x, y = pygame.mouse.get_pos()
            grid_x = int(x // tile_size)
            grid_y = int(y // tile_size)
            try:
                if level[grid_y, grid_x] != current_tile:
                    level[grid_y, grid_x] = current_tile
                    level_dirty = True
            except IndexError:
                pass

        if level_dirty:
            level_dirty = False

            if not waiting_for_result:
                task_queue.put(level.copy())
            waiting_for_result = True

        while not result_queue.empty():
            result = result_queue.get()
            if result is not None:
                status = convert_feature_to_status(result)
            waiting_for_result = False

        screen.fill((255, 255, 255))
        draw_map(screen, level)

        pygame.display.set_caption(instruction)

        draw_text(screen, f"I: {instruction}", (10, 10), size=30)
        draw_text(screen, f"C: {cond if cond is not None else 'No cond.'}", (10, 35), size=30)
        draw_text(screen, f"File Num {current_index + 1}/{len(files)}", (10, 60))

        short_path = join(*path.split("/")[-2:])
        draw_text(screen, f"File: {short_path}", (10, 80))
        shape_color = (255, 0, 0) if raw_level.shape != (16, 16) else (255, 255, 255)
        draw_text(screen, f"Level shape: {raw_level.shape}", (10, 100), color=shape_color)

        draw_text(screen, f"Key 1: Empty, 2: Wall, 3: Enemy, 5: Select index", (10, 450))
        draw_text(screen, f"Tile type: {current_tile}", (10, 470))
        draw_text(screen, "D: delete, S: save, R: reset", (10, 490))

        for i, s in enumerate(status):
            draw_text(screen, s, (10, 140 + i * 20), size=20)

        if waiting_for_result:
            draw_text(screen, "Validating...", (10, 120), size=20, color=(0, 0, 255))

        pygame.display.flip()
        pygame.time.Clock().tick(60)

    task_queue.put(None)
    loss_proc.join()
    pygame.quit()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    csv_file = os.path.abspath(os.path.join(dirname(abspath(__file__)), "..", "..", "..", "instruct", "sub_condition",
                                            "bert", "scn-1_se-whole.csv"))
    df = pd.read_csv(csv_file)
    parser.add_argument("--dir", type=str, default='levels')
    parser.add_argument("--task", type=str, default='all')
    config = parser.parse_args()
    inst_filter = df[df['reward_enum'] == int(config.task)]['instruction'].tolist() if config.task != 'all' else None
    print(f"Config: {config}")
    main(config, df, inst_filter)

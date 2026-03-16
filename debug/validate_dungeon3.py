import pygame
import hydra
import jax
from jax import numpy as jnp
import numpy as np
import sys
import matplotlib.pyplot as plt
import io

from conf.config import TrainConfig
from pcgrllm.validate_reward import read_file
from purejaxrl.experimental.s5.wrappers import LLMRewardWrapper
from utils import gymnax_pcgrl_make, init_config


def validate_dungeon3(enjoy_config: TrainConfig, episode_count: int = 1):
    enjoy_config = init_config(enjoy_config)

    enjoy_config.problem = 'dungeon3'
    enjoy_config.max_board_scans = 1
    # enjoy_config.representation = 'mazes'

    # enjoy_config = init_config_for_eval(enjoy_config)
    env, env_params = gymnax_pcgrl_make(enjoy_config.env_name, config=enjoy_config)
    env.prob.init_graphics()

    if hasattr(enjoy_config, 'reward_function_path') and enjoy_config.reward_function_path is not None:
        env = LLMRewardWrapper(env)

        reward_fn_str = read_file(enjoy_config.reward_function_path)

        exec_scope = {}
        exec(reward_fn_str, exec_scope)
        reward_fn = exec_scope['compute_reward']

        env.set_reward_fn(reward_fn)

    rng = jax.random.PRNGKey(enjoy_config.seed)
    obs, env_state = env.reset(rng)
    frame = env.render(env_state)  # Assuming env.render returns an RGB array
    frame_height, frame_width = frame.shape[:2]
    screen_height, screen_width = frame_height * 2, frame_width * 2

    pygame.init()

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("PCGRL Interactive Environment")
    clock = pygame.time.Clock()

    # Use a more aesthetic font
    pygame.font.init()
    font = pygame.font.Font(pygame.font.match_font("verdana", bold=True), 22)

    episode_length = env.max_steps  # Set an estimated maximum episode length

    # Define graph dimensions in pixels
    graph_width, graph_height = screen_width // 2, screen_height // 3

    for episode in range(episode_count):
        obs, env_state = env.reset(rng)

        done = False
        episode_reward = 0
        step_count = 0  # Initialize step count for the episode
        reward_history = []  # To store cumulative rewards per step

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Random sample action
            rng, subkey = jax.random.split(rng)
            action = env.sample_action(subkey)

            # If an action is selected, step the environment
            if action is not None:
                action = jnp.array(action).reshape(1, 1, 1)

                obs, env_state, reward, done, info = env.step(rng, env_state, action, env_params)
                episode_reward += reward
                step_count += 1  # Increment step count
                reward_history.append(episode_reward)
                print(f"Step: {step_count}, Action: {action}, Reward: {reward}, Done: {done}")

            # Render the environment to a frame
            frame = env.render(env_state)  # Assuming env.render returns an RGB array
            frame = jax.device_get(frame)  # Converts JAX Array to NumPy array
            frame = frame[..., :3]

            # Convert to uint8 if necessary
            frame = np.clip(frame, 0, 255).astype(np.uint8)

            # Display the frame in Pygame
            pygame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            screen.blit(pygame.transform.scale(pygame_surface, (screen_width, screen_height)), (0, 0))

            # Render text
            step_text = font.render(f"Step: {step_count}", True, (255, 255, 255))
            action_text = font.render(f"Action: {action}", True, (255, 255, 255))
            reward_text = font.render(f"Reward: {reward:.2f}", True, (255, 255, 255))
            episode_reward_text = font.render(f"Episode Reward: {episode_reward:.2f}", True, (255, 255, 255))

            # Blit text onto the screen
            screen.blit(step_text, (10, 10))
            screen.blit(action_text, (10, 40))
            screen.blit(reward_text, (10, 70))
            screen.blit(episode_reward_text, (10, 100))

            # Plot and display cumulative reward graph
            if len(reward_history) > 1:
                # Adjust figsize to match Pygame graph dimensions
                fig, ax = plt.subplots(figsize=(graph_width / 100, graph_height / 100), dpi=100)
                ax.plot(range(1, len(reward_history) + 1), reward_history, color="blue", linewidth=2)
                ax.set_xlim(1, episode_length)  # Set x-axis limit to the estimated episode length
                ax.tick_params(axis='both', labelsize=6)
                ax.grid(True)

                # Save the plot to a surface for Pygame
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)  # Save at 100 DPI
                buf.seek(0)
                graph_surface = pygame.image.load(buf)
                buf.close()
                plt.close(fig)

                # Display the graph in the bottom left corner
                graph_x, graph_y = 10, screen_height - graph_height - 10
                screen.blit(graph_surface, (graph_x, graph_y))

            pygame.display.flip()

            clock.tick(30)  # Limit FPS to 30

    pygame.quit()

    return episode_reward


if __name__ == '__main__':
    validate_dungeon3(init_config(TrainConfig()))
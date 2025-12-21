"""Rymdboard: A tile-placement game environment for PufferLib Ocean.

The goal is to connect crystal nodes on a 10x10 grid by placing
streets and extraction buildings from a market of 7 cards.
Win condition: Connect 2+ crystal nodes with extraction buildings.

Reference: research/simple_env.py, research/simple_game_engine.py
"""

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.rymdboard import binding


# Observation dimensions (must match rymdboard.h)
BOARD_SIZE = 10
BOARD_TILES = BOARD_SIZE * BOARD_SIZE
OBS_BOARD_SIZE = 100
OBS_MARKET_SIZE = 210  # 7 cards * 30 features each
OBS_CRYSTAL_SIZE = 12
OBS_TOTAL_SIZE = OBS_BOARD_SIZE + OBS_MARKET_SIZE + OBS_CRYSTAL_SIZE

# Action space: 7 cards × 10 × 10 positions × 4 rotations = 2800
NUM_ACTIONS = 2800


class Rymdboard(pufferlib.PufferEnv):
    """Rymdboard environment using C backend for fast vectorized simulation."""
    
    def __init__(
        self,
        num_envs=1,
        render_mode=None,
        log_interval=128,
        max_steps=200,
        buf=None,
        frameskip=1,
        seed=0,
    ):
        """Initialize Rymdboard environment.
        
        Args:
            num_envs: Number of parallel environments
            render_mode: Rendering mode (not implemented yet)
            log_interval: How often to aggregate and return logs
            max_steps: Maximum steps per episode before truncation
            buf: Pre-allocated buffer (optional)
            seed: Random seed
        """
        # Define observation and action spaces
        self.single_observation_space = gymnasium.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_TOTAL_SIZE + NUM_ACTIONS,),
            dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(NUM_ACTIONS)
        
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.max_steps = max_steps
        self.frameskip = frameskip
        self.frameskip = frameskip
        
        if render_mode is None:
            import sys
            if '--render-mode' in sys.argv:
                try:
                    idx = sys.argv.index('--render-mode')
                    if idx + 1 < len(sys.argv):
                        render_mode = sys.argv[idx + 1]
                        self.render_mode = render_mode
                        print(f"DEBUG: Manually parsed render_mode={render_mode} from sys.argv")
                except ValueError:
                    pass

        super().__init__(buf)
        
        # Initialize C environments
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            max_steps=max_steps,
            frameskip=frameskip,
            render_mode=1 if render_mode == 'human' else 0,
        )
    
    def reset(self, seed=0):
        """Reset all environments."""
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        if self.render_mode == 'human' or self.render_mode == 1:
            self.render()
        return self.observations, []
    
    def step(self, actions):
        """Step all environments with the given actions."""
        self.tick += 1
        
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        
        info = []
        if self.tick % self.log_interval == 0:
            log_info = binding.vec_log(self.c_envs)
            if log_info:  # May be empty dict if no episodes completed
                info.append(log_info)

        if self.render_mode == 'human' or self.render_mode == 1:
            self.render()
        
        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info,
        )
    
    def render(self):
        """Render the first environment."""
        binding.vec_render(self.c_envs, 0)
    
    def close(self):
        """Clean up C environments."""
        binding.vec_close(self.c_envs)


# For compatibility with env_creator in environment.py
def make_rymdboard(num_envs=1, render_mode=None, log_interval=128,
                   max_steps=200, buf=None, seed=0, frameskip=1, **kwargs):
    """Factory function to create Rymdboard environment."""
    return Rymdboard(
        num_envs=num_envs,
        render_mode=render_mode,
        log_interval=log_interval,
        max_steps=max_steps,
        buf=buf,
        seed=seed,
        frameskip=frameskip,
    )


if __name__ == '__main__':
    """Performance benchmark."""
    import time
    
    N = 4096
    env = Rymdboard(num_envs=N)
    env.reset()
    steps = 0
    
    CACHE = 1024
    actions = np.random.randint(0, NUM_ACTIONS, (CACHE, N))
    
    i = 0
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += N
        i += 1
    
    print(f'Rymdboard SPS: {int(steps / (time.time() - start))}')


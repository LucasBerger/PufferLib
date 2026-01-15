"""Quarryboard: A competitive 4-player environment with Scarcity & Quarry mechanics.

Goal: Extract most resources (Crystals) to win.
"""

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.quarryboard import binding

class Quarryboard(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        render_mode=None,
        log_interval=128,
        max_steps=200,
        buf=None,
        frameskip=1,
        seed=0,
        board_size=30,
        start_crystals=5,
        start_stones=5,
    ):
        self.board_size = board_size
        self.start_crystals = start_crystals
        self.start_stones = start_stones
        
        # Action Space: 7 cards * W * H * 4 rotations + Pass + Finish
        self.num_std_actions = 7 * board_size * board_size * 4
        self.num_actions = self.num_std_actions + 2 # Pass, Finish
        
        # Observation Space
        self.obs_board_size = board_size * board_size
        self.obs_market_size = 210
        self.obs_crystal_size = 70 
        self.obs_resource_size = 20 # 4 players * 5 stats (score, stone, cumulative, finished, dead)
        self.obs_sync_size = 3 
        
        self.obs_real_size = (self.obs_board_size + self.obs_market_size + 
                             self.obs_crystal_size + self.obs_resource_size + self.obs_sync_size)
        
        # Total = Real + Action Mask
        self.obs_total_size = self.obs_real_size + self.num_actions
        
        self.single_observation_space = gymnasium.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_total_size,),
            dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(self.num_actions)
        
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.max_steps = max_steps
        self.frameskip = frameskip
        
        super().__init__(buf)
        
        self.num_agents = num_envs
        
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            max_steps=max_steps,
            frameskip=self.frameskip,
            render_mode=1 if self.render_mode == 'human' else (2 if self.render_mode == 'cli' else 0),
            board_size=board_size,
            start_crystals=start_crystals,
            start_stones=start_stones,
        )
    
    def render(self):
        binding.vec_render(self.c_envs, 0)
    
    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        if self.render_mode == 'human':
            self.render()
        return self.observations, []
    
    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        
        info = []
        if self.tick % self.log_interval == 0:
            log_info = binding.vec_log(self.c_envs)
            if log_info:
                info.append(log_info)

        if self.render_mode == 'human':
            self.render()

        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info,
        )
    
    def close(self):
        binding.vec_close(self.c_envs)

def make_quarryboard(num_envs=1, render_mode=None, log_interval=128,
                   max_steps=200, buf=None, seed=0, frameskip=1, board_size=30, 
                   start_crystals=5, start_stones=5, **kwargs):
    return Quarryboard(
        num_envs=num_envs,
        render_mode=render_mode,
        log_interval=log_interval,
        max_steps=max_steps,
        buf=buf,
        seed=seed,
        frameskip=frameskip,
        board_size=board_size,
        start_crystals=start_crystals,
        start_stones=start_stones,
    )

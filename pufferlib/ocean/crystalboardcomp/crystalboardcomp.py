"""CrystalboardComp: A competitive 4-player environment.

Goal: Extract most resources.
"""

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.crystalboardcomp import binding

class CrystalboardComp(pufferlib.PufferEnv):
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
    ):
        self.board_size = board_size
        
        # Action Space: 7 cards * W * H * 4 rotations
        self.num_actions = 7 * board_size * board_size * 4
        
        # Observation Space
        self.obs_board_size = board_size * board_size
        self.obs_market_size = 210
        self.obs_crystal_size = 70 # 10 crystals * 7 features (x,y,c1,c2,c3,c4,res)
        self.obs_sync_size = 3 # current_player, human_ready, human_action
        
        self.obs_real_size = self.obs_board_size + self.obs_market_size + self.obs_crystal_size + self.obs_sync_size
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
        
        # C environment is turn-based, exposing 1 agent interface at a time
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
            render_mode=1 if self.render_mode == 'human' else 0,
            board_size=board_size,
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

def make_crystalboardcomp(num_envs=1, render_mode=None, log_interval=128,
                   max_steps=200, buf=None, seed=0, frameskip=1, board_size=30, **kwargs):
    return CrystalboardComp(
        num_envs=num_envs,
        render_mode=render_mode,
        log_interval=log_interval,
        max_steps=max_steps,
        buf=buf,
        seed=seed,
        frameskip=frameskip,
        board_size=board_size,
    )

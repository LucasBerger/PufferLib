import numpy as np
import torch
import time
import pufferlib
from pufferlib.ocean.quarryboard import quarryboard
from pufferlib.ocean.torch import QuarryboardUNet

def eval_quarryboard(
    checkpoint_paths=[None, None, None, None], 
    human_players=[0], # List of player indices (0-3) controlled by human
    board_size=30,
    frameskip=1,
    render_mode='human'  # 'human' for GUI, 'cli' for text output
):
    # Initialize Environment
    # Note: We use render_mode='human' to trigger the window creation in C
    # Or render_mode='cli' for text-based output
    rm = 1 if render_mode == 'human' else (2 if render_mode == 'cli' else 0)
    
    env = quarryboard.Quarryboard(
        num_envs=1, 
        render_mode=render_mode, 
        board_size=board_size,
        frameskip=frameskip,
        start_crystals=5,
        start_stones=5
    )
    env.reset()
    
    # Load Policies
    policies = []
    for i, path in enumerate(checkpoint_paths):
        if i in human_players:
            policies.append("HUMAN")
            print(f"Player {i+1}: HUMAN")
        elif path:
            # Load Checkpoint
            # Assuming standard PufferLib policy structure
            policy_model = QuarryboardUNet(env, board_size=board_size)
            
            # Load state dict
            try:
                ckpt = torch.load(path, map_location='cpu')
                # Handle different checkpoint formats (PufferLib vs CleanRL style)
                if 'agent_state_dict' in ckpt:
                    state_dict = ckpt['agent_state_dict']
                elif 'model_state_dict' in ckpt:
                    state_dict = ckpt['model_state_dict']
                else:
                    state_dict = ckpt
                    
                # Clean prefix if needed (e.g. "module.")
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                # --- TARGETED WEIGHT SURGERY ---
                key = 'owner_embed.weight'
                if key in state_dict:
                    saved_weight = state_dict[key]
                    current_weight = policy_model.state_dict()[key]
                    if saved_weight.shape != current_weight.shape:
                        print(f"Reshaping {key}: {saved_weight.shape} -> {current_weight.shape}")
                        new_weight = current_weight.clone()
                        rows_to_copy = min(saved_weight.shape[0], current_weight.shape[0])
                        cols_to_copy = min(saved_weight.shape[1], current_weight.shape[1])
                        new_weight[:rows_to_copy, :cols_to_copy] = saved_weight[:rows_to_copy, :cols_to_copy]
                        state_dict[key] = new_weight

                # Load
                policy_model.load_state_dict(state_dict, strict=False)
                policy_model.eval()
                policies.append(policy_model)
                print(f"Player {i+1}: AI (Loaded from {path})")
            except Exception as e:
                print(f"Failed to load checkpoint for Player {i+1}: {e}")
                policies.append("RANDOM")
        else:
            policies.append("RANDOM")
            print(f"Player {i+1}: RANDOM")

    # Game Loop
    done = False
    
    # Calculate offsets for manual reading
    # Board (900) + Market (210) + Crystals (70) + Resources (20)
    # Sync Info starts after this
    OBS_SYNC_OFFSET = 30*30 + 210 + 70 + 20
    
    while not done:
        env.render()
        
        action_to_take = None
        
        tick = 0
        while action_to_take is None:
            env.render()
            time.sleep(0.02) # Throttle to ~50 FPS
            tick += 1
            
            # Read Sync Info directly from observation buffer
            obs_view = env.observations[0]
            
            current_player_idx = int(obs_view[OBS_SYNC_OFFSET])
            human_ready = obs_view[OBS_SYNC_OFFSET + 1]
            last_human_action = int(obs_view[OBS_SYNC_OFFSET + 2])
            
            if current_player_idx < 0 or current_player_idx >= 4:
                continue
                
            current_policy = policies[current_player_idx]
            
            if current_policy == "HUMAN":
                print("Human ready: ", human_ready)
                print("Last human action: ", last_human_action)
                if human_ready > 0.5:
                    action_to_take = last_human_action
            else:
                # AI Turn
                obs_full = torch.as_tensor(env.observations)
                
                if current_policy == "RANDOM":
                    if isinstance(env.observations, tuple):
                        observations, action_mask = env.observations
                    elif env.observations.shape[-1] > env.obs_real_size:
                        action_mask = env.observations[:, env.obs_real_size:]
                    else:
                        action_mask = None # Should not happen unless obs shape changed

                    if action_mask is not None:
                        action_mask = action_mask.astype(np.int8)
                        # Ensure mask shape is correct for multiple envs
                        if len(action_mask.shape) == 1:
                            action_mask = action_mask.reshape(1, -1)
                        action_to_take = env.action_space.sample(mask=tuple(action_mask))[0]
                    else:
                        action_to_take = env.action_space.sample()[0]
                else:
                    with torch.no_grad():
                         # Ensure batch dim
                         if len(obs_full.shape) == 1: obs_full = obs_full.unsqueeze(0)
                         actions, _ = current_policy(obs_full)
                         action_to_take = torch.argmax(actions[0]).item()
        
        # 3. Step
        o, r, d, t, i = env.step([action_to_take])
        done = d[0] or t[0]
        
        time.sleep(0.05)
        
    print("Game Over")
    env.close()
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run quarryboard evaluation with different render modes')
    parser.add_argument('--render-mode', type=str, default='cli', choices=['cli', 'human'],
                        help='Render mode: cli for text output, human for GUI (default: cli)')
    args = parser.parse_args()
    
    # Example: All Random with selected render mode
    print(f"Running quarryboard with {args.render_mode} render mode...")
    if args.render_mode == 'cli':
        print("This will show text output of game state.")
    else:
        print("This will open a GUI window.")
    print()
    
    eval_quarryboard(
        checkpoint_paths=[None, None, None, None], 
        human_players=[0],  # No human players - all random
        render_mode=args.render_mode
    )

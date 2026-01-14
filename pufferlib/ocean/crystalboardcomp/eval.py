import numpy as np
import torch
import time
import pufferlib
from pufferlib.ocean.crystalboardcomp import crystalboardcomp, binding
from pufferlib.ocean.torch import CrystalboardCompUNet

def eval_crystalboard(
    checkpoint_paths=[None, None, None, None], 
    human_players=[0], # List of player indices (0-3) controlled by human
    board_size=30,
    frameskip=1
):
    # Initialize Environment
    # Note: We use render_mode='human' to trigger the window creation in C
    env = crystalboardcomp.CrystalboardComp(
        num_envs=1, 
        render_mode=None, 
        board_size=board_size,
        frameskip=frameskip
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
            # We need to re-create the policy architecture first
            policy_model = CrystalboardCompUNet(env, board_size=board_size)
            
            # Load state dict
            # The checkpoint is usually a full trainer state or just model state
            # Let's assume standard torch.load
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
                    # Get the current weight from the model to check the required shape
                    current_weight = policy_model.state_dict()[key]
                    
                    if saved_weight.shape != current_weight.shape:
                        print(f"Reshaping {key}: {saved_weight.shape} -> {current_weight.shape}")
                        
                        # Create a buffer of the correct size (initialized like the current model)
                        new_weight = current_weight.clone()
                        
                        # Determine how many rows/columns we can actually copy
                        # (This handles both growing and shrinking the embedding)
                        rows_to_copy = min(saved_weight.shape[0], current_weight.shape[0])
                        cols_to_copy = min(saved_weight.shape[1], current_weight.shape[1])
                        
                        # Copy the overlapping part of the weights
                        new_weight[:rows_to_copy, :cols_to_copy] = saved_weight[:rows_to_copy, :cols_to_copy]
                        
                        # Update the state_dict with our "patched" version
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
        
    # Calculate offsets for manual reading
    # Board (900) + Market (210) + Crystals (70)
    # Sync Info starts at 1180
    OBS_SYNC_OFFSET = 30*30 + 210 + 70
    
    while not done:
        # Explicit rendering at start of loop to update Sync Bytes and window
        env.render()
        
        action_to_take = None
        
        tick = 0
        while action_to_take is None:
            env.render()
            time.sleep(0.02) # Throttle to ~50 FPS to avoid hanging the window
            tick += 1
            
            # Read Sync Info directly from observation buffer
            # We must use the first part of the buffer (Agent 0 / Env 0 view)
            # Since C updates this in-place in c_render, it should be fresh.
            obs_view = env.observations[0]
            
            current_player_idx = int(obs_view[OBS_SYNC_OFFSET])
            human_ready = obs_view[OBS_SYNC_OFFSET + 1]
            last_human_action = int(obs_view[OBS_SYNC_OFFSET + 2])
            
            # Sanity check range
            if current_player_idx < 0 or current_player_idx >= 4:
                # Should not happen, but safeguard
                continue
                
            current_policy = policies[current_player_idx]
            
            if current_policy == "HUMAN":
                if human_ready > 0.5:
                    action_to_take = last_human_action
            else:
                # AI Turn
                # Passing raw obs mimics PufferEnv check logic best.
                obs_full = torch.as_tensor(env.observations)
                
                # Explicit render for AI/Random turns
                # env.render() # Implicit in step if render_mode=human
                # time.sleep(0.02)
                
                if current_policy == "RANDOM":
                    if isinstance(env.observations, tuple):
                        observations, action_mask = env.observations
                    elif env.observations.shape[-1] > env.obs_real_size:
                        action_mask = env.observations[:, env.obs_real_size:]
                    else:
                        action_mask = None

                    action_mask = action_mask.astype(np.int8)
                    action_to_take = env.action_space.sample(mask=tuple(action_mask))[0]
                else:
                    with torch.no_grad():
                         actions, _ = current_policy(obs_full)
                         action_to_take = torch.argmax(actions[0]).item()
        
        # 3. Step
        o, r, d, t, i = env.step([action_to_take])
        obs_view = env.observations[0] # Refresh view just in case pointer changed (unlikely but safe)
        done = d[0] or t[0]
        
        # Throttle the entire loop to reasonable FPS (e.g. 20 FPS)
        # This prevents AI/Random from spinning 1000fps and hanging window
        time.sleep(0.05)
        
    print("Game Over")
    # Close window
    env.close()
    
if __name__ == "__main__":
    # Example: P1 Human, others Random
    eval_crystalboard(checkpoint_paths=[None, None, "./experiments/puffer_crystalboardcomp_BOAR-383/model_puffer_crystalboardcomp_000430.pt", "./experiments/puffer_crystalboardcomp_BOAR-383/model_puffer_crystalboardcomp_000430.pt"], human_players=[0])

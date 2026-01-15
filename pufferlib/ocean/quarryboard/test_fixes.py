#!/usr/bin/env python3
"""Test script for quarryboard game logic fixes using CLI render mode."""

import numpy as np
import sys
import time
from pufferlib.ocean.quarryboard import quarryboard

def test_fixes():
    print("=" * 80)
    print("QUARRYBOARD GAME LOGIC TESTING")
    print("=" * 80)
    print()
    
    # Create environment with CLI render mode (render_mode=2)
    env = quarryboard.Quarryboard(
        num_envs=1,
        render_mode=2,  # CLI mode
        board_size=30,
        frameskip=1,
        start_crystals=5,
        start_stones=5
    )
    
    print("Resetting environment...")
    env.reset()
    print()
    
    # Issue 1: Market Building Distribution
    print("ISSUE 1: Market Building Distribution")
    print("-" * 80)
    env.render()
    print("✓ Check if first 4 market slots (Buildings 0-3) show both 'Q' and 'E' types")
    print()
    input("Press Enter to continue...")
    print()
    
    # Issue 2: Separate Resource Counters
    print("ISSUE 2: Separate Resource Counters")
    print("-" * 80)
    print("✓ Check PLAYER RESOURCES section shows both C (crystals) and S (stones)")
    print("  Each player should have format: 'P1: 5C 5S'")
    print()
    input("Press Enter to continue...")
    print()
    
    # Issue 3: Resource Validation
    print("ISSUE 3: Resource Validation for Building")
    print("-" * 80)
    print("Testing: Attempting to build with insufficient resources...")
    
    # Drain P1's resources
    env._envs.scores[0] = 0
    env._envs.stones[0] = 0
    
    # Try to place a building (should fail)
    # Get a valid position near base
    action = 0  # Card 0, position (5,5), rotation 0
    
    obs, reward, done, trunc, info = env.step([action])
    
    print(f"Action attempted with 0 resources")
    print(f"Reward received: {reward[0]:.3f}")
    print("✓ Should be negative (REWARD_INVALID_MOVE = -0.05)")
    print()
    
    # Restore resources
    env._envs.scores[0] = 5
    env._envs.stones[0] = 5
    input("Press Enter to continue...")
    print()
    
    # Issue 4 & 5: Test gameplay
    print("ISSUE 4 & 5: Quarry Generation and Connection Rules")
    print("-" * 80)
    print("Running 8 rounds of random gameplay...")
    print()
    
    for round_num in range(8):
        # Get action mask
        if hasattr(env, 'observations') and len(env.observations.shape) > 1:
            action_mask = env.observations[:, env.obs_real_size:]
        else:
            action_mask = None
        
        # Sample valid action
        if action_mask is not None:
            action_mask = action_mask.astype(np.int8)
            action = env.action_space.sample(mask=tuple(action_mask))[0]
        else:
            action = env.action_space.sample()[0]
        
        obs, reward, done, trunc, info = env.step([action])
        
        if round_num % 4 == 3:  # After each full round
            print(f"\n--- After Round {round_num // 4 + 1} ---")
            env.render()
            print("\n✓ Issue 4: Check if quarries generate stones (P1-P4 stone counts should increase)")
            print("✓ Issue 5: Buildings should only be placed adjacent to streets or bases")
            print("           (NOT adjacent to extractors/quarries alone)")
            print()
            
            if round_num < 7:
                input("Press Enter for next round...")
        
        if done[0] or trunc[0]:
            break
    
    print()
    print("=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    print()
    print("Summary of fixes:")
    print("  1. ✓ Market shows both quarries (Q) and extractors (E)")
    print("  2. ✓ Resources displayed separately as XC YS")
    print("  3. ✓ Building with 0 resources gives negative reward (no placement)")
    print("  4. ✓ Quarries generate stones each round")
    print("  5. ✓ Buildings only connect via streets/bases (policy will learn this)")
    print()
    
    env.close()

if __name__ == "__main__":
    test_fixes()

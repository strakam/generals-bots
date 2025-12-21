"""
Test correctness of JAX environment vs NumPy environment.
Ensures both implementations produce identical results using shared grid factory.
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp

from generals.envs import GymnasiumGenerals
from generals.envs.jax_env import VectorizedJaxEnv
from generals import GridFactory
from generals.core.action import Action
from generals.core import game_jax
from generals.envs.jax_rendering_adapter import JaxGameAdapter
import pygame


def replay_frames_jax(env, frames):
    """Interactive replay of captured JAX frames with arrow key navigation."""
    if not frames:
        return
    
    # Initialize GUI if not already done
    if env.gui is None:
        from generals.gui import GUI
        from generals.gui.properties import GuiMode
        
        # Create adapter for first frame to initialize GUI
        first_frame = frames[0]
        first_info = game_jax.get_info(first_frame['state'])
        
        adapted_game = JaxGameAdapter(
            first_frame['state'],
            env.agent_names,
            first_info
        )
        
        env.gui = GUI(
            adapted_game,
            env.agent_data,
            GuiMode.TRAIN,
            env.speed_multiplier
        )
    
    print("\n=== REPLAY MODE (JAX) ===")
    print("Controls: LEFT/RIGHT arrows to navigate, ESC/Q to exit")
    
    current_frame = 0
    running = True
    clock = pygame.time.Clock()
    
    while running:
        # Handle input FIRST
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_LEFT:
                    current_frame = max(0, current_frame - 1)
                elif event.key == pygame.K_RIGHT:
                    current_frame = min(len(frames) - 1, current_frame + 1)
                elif event.key == pygame.K_HOME:
                    current_frame = 0
                elif event.key == pygame.K_END:
                    current_frame = len(frames) - 1
        
        if not running:
            break
            
        # Display current frame
        frame = frames[current_frame]
        frame_info = game_jax.get_info(frame['state'])
        
        # Update the GUI's game adapter with the current frame's state
        env.gui.properties.game.update_from_state(
            frame['state'],
            frame_info
        )
        
        # Show frame info in title
        pygame.display.set_caption(f"Generals JAX - Replay Frame {current_frame + 1}/{len(frames)} (Step {frame['step']})")
        
        # Render the frame
        env.gui._GUI__renderer.render(fps=10)
        clock.tick(30)  # 30 FPS for smooth navigation
    
    print("\n=== REPLAY ENDED ===\n")


def replay_frames_numpy(env, frames):
    """Interactive replay of captured NumPy frames with arrow key navigation."""
    if not frames:
        return
    
    # Initialize GUI if not already done
    if env.gui is None:
        from generals.gui import GUI
        from generals.gui.properties import GuiMode
        
        # Render first frame to initialize GUI
        env.render()
    
    print("\n=== REPLAY MODE (NumPy) ===")
    print("Controls: LEFT/RIGHT arrows to navigate, ESC/Q to exit")
    
    current_frame = 0
    running = True
    clock = pygame.time.Clock()
    
    while running:
        # Handle input FIRST
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_LEFT:
                    current_frame = max(0, current_frame - 1)
                elif event.key == pygame.K_RIGHT:
                    current_frame = min(len(frames) - 1, current_frame + 1)
                elif event.key == pygame.K_HOME:
                    current_frame = 0
                elif event.key == pygame.K_END:
                    current_frame = len(frames) - 1
        
        if not running:
            break
            
        # Display current frame - update NumPy game state
        frame = frames[current_frame]
        env.game.channels = frame['channels']
        env.game.time = frame['time']
        
        # Show frame info in title
        pygame.display.set_caption(f"Generals NumPy - Replay Frame {current_frame + 1}/{len(frames)} (Step {frame['step']})")
        
        # Render the frame
        env.gui._GUI__renderer.render(fps=10)
        clock.tick(30)  # 30 FPS for smooth navigation
    
    print("\n=== REPLAY ENDED ===\n")


def test_single_game(game_num, seed, num_steps, grid_width, grid_height, render_mode='jax'):
    """Test a single game for correctness."""
    
    # Create grid factories with same seed for both environments
    grid_factory_np = GridFactory(
        min_grid_dims=(grid_height, grid_width),
        max_grid_dims=(grid_height, grid_width),
        mountain_density=0.2,
        city_density=0.05,
        general_positions=[[2, 2], [grid_height - 1, grid_width - 1]],
        seed=seed
    )
    
    grid_factory_jax = GridFactory(
        min_grid_dims=(grid_height, grid_width),
        max_grid_dims=(grid_height, grid_width),
        mountain_density=0.2,
        city_density=0.05,
        general_positions=[[2, 2], [grid_height - 1, grid_width - 1]],
        seed=seed  # Same seed ensures same grid generation
    )
    
    # Create NumPy environment - NOT wrapped in vector env
    # GymnasiumGenerals already handles multiple players internally
    np_env = GymnasiumGenerals(
        agents=["Player1", "Player2"],
        grid_factory=grid_factory_np,
        truncation=5000,
        pad_observations_to=grid_height,  # Match grid size
        render_mode="human" if render_mode == 'numpy' else None,
    )
    
    jax_env = VectorizedJaxEnv(
        num_envs=1,
        grid_size=(grid_height, grid_width),
        grid_factory=grid_factory_jax,
        render_mode="human" if render_mode == 'jax' else None,
    )
    
    # Reset both environments
    np_obs_array, np_info = np_env.reset(seed=seed)
    jax_obs, jax_info = jax_env.reset(seed=seed)
    
    success = True
    frames = []  # Store frames for replay on failure
    
    # Run steps with same actions (less verbose)
    for step in range(num_steps):
        # Generate deterministic actions
        np.random.seed(seed + step)
        
        # Get NumPy action masks from info
        np_masks = np_info.get('action_mask', None)  # Shape: (2, num_actions)
        
        # Generate masked random actions for both players
        actions_np = []
        actions_jax_list = []
        
        for player_idx in range(2):
            # Get valid actions for this player using NumPy mask
            if np_masks is not None:
                valid_actions = np.where(np_masks[player_idx])[0]
            else:
                valid_actions = np.arange(grid_height * grid_width * 4 * 2 + 1)
            
            if len(valid_actions) == 0:
                # No valid actions, pass
                action_np = Action(to_pass=True)
                action_jax = [1, 0, 0, 0, 0]
            else:
                # Sample a random valid action
                action_idx = np.random.choice(valid_actions)
                
                if action_idx == 0:  # Pass action
                    action_np = Action(to_pass=True)
                    action_jax = [1, 0, 0, 0, 0]
                else:
                    # Decode action index to (row, col, direction, split)
                    action_idx -= 1  # Remove pass action offset
                    split = action_idx % 2
                    action_idx //= 2
                    direction = action_idx % 4
                    action_idx //= 4
                    col = action_idx % grid_width
                    row = action_idx // grid_width
                    
                    action_np = Action(to_pass=False, row=row, col=col, direction=direction, to_split=bool(split))
                    action_jax = [0, row, col, direction, split]
            
            actions_np.append(action_np)
            actions_jax_list.append(action_jax)
        
        actions_jax = jnp.array([actions_jax_list], dtype=jnp.int32)  # Shape: (1, 2, 5)
        
        # Debug: print all moves with proper formatting
        direction_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        for player_idx, action_np in enumerate(actions_np):
            if action_np[0] == 0:  # not pass
                row, col = action_np[1], action_np[2]
                direction = direction_names[action_np[3]]
                split = 'SPLIT' if action_np[4] == 1 else 'ALL'
                army_before = np_env.game.channels.armies[row, col]
        
        # Step both environments
        np_obs_array, np_reward, np_terminated, np_truncated, np_info = np_env.step(actions_np)
        jax_obs, jax_rewards, jax_terminated, jax_truncated, jax_info = jax_env.step(actions_jax)
        
        # Capture frames for replay
        import copy
        frames.append({
            # JAX state
            'jax_obs': jax_obs,
            'state': jax.tree.map(lambda x: x[0], jax_env.states),  # Get first env's state
            # NumPy state
            'channels': copy.deepcopy(np_env.game.channels),
            'time': np_env.game.time,
            # Common
            'step': step,
            'actions': actions_jax_list,
        })
        
        # Compare terminated/truncated FIRST before comparing observations
        np_terminated_val = bool(np_terminated)
        np_truncated_val = bool(np_truncated)
        jax_terminated_val = bool(jax_terminated[0])
        jax_truncated_val = bool(jax_truncated[0])
        
        if np_terminated_val != jax_terminated_val:
            print(f"\n❌ Game {game_num} FAILED at step {step}")
            print(f"   Terminated differs: numpy={np_terminated_val}, jax={jax_terminated_val}")
            if render_mode == 'numpy':
                replay_frames_numpy(np_env, frames)
            else:
                replay_frames_jax(jax_env, frames)
            return False
            
        if np_truncated_val != jax_truncated_val:
            print(f"\n❌ Game {game_num} FAILED at step {step}")
            print(f"   Truncated differs: numpy={np_truncated_val}, jax={jax_truncated_val}")
            if render_mode == 'numpy':
                replay_frames_numpy(np_env, frames)
            else:
                replay_frames_jax(jax_env, frames)
            return False
        
        # Check if terminated/truncated - stop test here since NumPy doesn't auto-reset
        if np_terminated_val or np_truncated_val:
            print(f"   Both envs terminated/truncated at step {step} - test complete")
            return True  # Success - game completed
        
        # Compare observations (NumPy: (2,15,h,w), JAX: (1,2,15,h,w))
        np_obs_tensor = np_obs_array
        jax_obs_tensor = np.array(jax_obs.as_tensor()[0])
        
        # Compare observations channel by channel
        channel_names = [
            "armies", "generals", "cities", "mountains", "neutral", "owned",
            "opponent_cells", "fog", "structures_in_fog", "owned_land_count",
            "owned_army_count", "opponent_land_count", "opponent_army_count",
            "timestep", "priority"
        ]
        
        obs_match = True
        for player_idx in range(2):
            for ch_idx in range(15):
                np_channel = np_obs_tensor[player_idx, ch_idx]
                jax_channel = jax_obs_tensor[player_idx, ch_idx]
                
                if not np.allclose(np_channel, jax_channel, rtol=1e-5, atol=1e-5):
                    print(f"\n❌ Game {game_num} FAILED at step {step}")
                    print(f"   Player {player_idx}, Channel {ch_idx} ({channel_names[ch_idx]}) differs!")
                    print(f"   Max diff: {np.abs(np_channel - jax_channel).max()}")
                    print(f"\n   NumPy channel:")
                    with np.printoptions(threshold=np.inf, linewidth=200):
                        print(np_channel.astype(int))
                    print(f"\n   JAX channel:")
                    with np.printoptions(threshold=np.inf, linewidth=200):
                        print(jax_channel.astype(int))
                    obs_match = False
                    success = False
                    if render_mode == 'numpy':
                        replay_frames_numpy(np_env, frames)
                    else:
                        replay_frames_jax(jax_env, frames)
                    break
            
            if not obs_match:
                break
        
        if not obs_match:
            break
        
        # Already checked termination above, no need to check again here
        
    return True


def test_numpy_vs_jax(render_mode='jax'):
    """Run multiple games to test correctness."""
    print("\n=== Testing NumPy vs JAX Equivalence ===")
    print(f"Rendering mode: {render_mode}")
    
    # Debug game 9 specifically
    num_games = 50
    num_steps = 2000
    grid_width = 4
    grid_height = 4
    base_seed = 42
    
    passed = 0
    failed = 0
    
    for game_num in range(num_games):
        seed = base_seed + game_num * 1000
        result = test_single_game(game_num + 1, seed, num_steps, grid_width, grid_height, render_mode)
        
        if result:
            print(f"✅ Game {game_num + 1}/{num_games} PASSED")
            passed += 1
        else:
            failed += 1
            return
    
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{num_games} games passed")
    if failed == 0:
        print("✅ All games passed! Implementations are equivalent.")
    else:
        print(f"❌ {failed} games failed!")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test NumPy vs JAX environment correctness')
    parser.add_argument('--numpy', action='store_true', help='Render NumPy environment')
    parser.add_argument('--jax', action='store_true', help='Render JAX environment (default)')
    args = parser.parse_args()
    
    # Determine render mode
    if args.numpy:
        render_mode = 'numpy'
    else:
        render_mode = 'jax'  # Default to JAX
    
    test_numpy_vs_jax(render_mode)

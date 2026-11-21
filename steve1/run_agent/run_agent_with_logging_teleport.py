import argparse
import csv
import inspect
import os
import shlex
import sys
import zipfile
from datetime import datetime
from typing import Any, Dict

import cv2
import numpy as np
import torch
from tqdm import tqdm

import globals
from multisteve.mc_creative import place_tree_commands
from steve1.config import PRIOR_INFO, DEVICE
from steve1.data.text_alignment.vae import load_vae_model
from steve1.run_agent.paper_prompts import load_visual_prompt_embeds
from steve1.run_agent.programmatic_eval import ProgrammaticEvaluator
from steve1.utils.embed_utils import get_prior_embed
from steve1.utils.mineclip_agent_env_utils import load_mineclip_agent_env, load_mineclip_wconfig, make_agent
from steve1.utils.video_utils import save_frames_as_video

FPS = 20

PLACEMENT_DEFAULTS: Dict[str, Any] = {
    'x': 4,
    'y': 0,
    'z': 7,
    'tree_type': 'oak',
    'relative': True,
    'villager': True,
    'villager_profession': 'unemployed',
    'villager_type': 'plains',
    'logs_above_villager': False,
}

SIDE_DEFAULT_COORDS = {
    'left': (4, 0, 7),
    'right': (-4, 0, 7),
}


class FakeEnvironment:
    def __init__(self, zip_filepath):
        self.zip_filepath = zip_filepath
        self.frames_dir = None
        self.frames = []
        self.current_frame_index = 0

    def reset(self):
        import tempfile
        import shutil
        self.current_frame_index = 0
        if self.frames_dir:
            shutil.rmtree(self.frames_dir)
        self.frames_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(self.zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(self.frames_dir)
        self.frames = sorted([os.path.join(self.frames_dir, f) for f in os.listdir(self.frames_dir) if f.endswith('.png')])
        return self._get_obs()

    def step(self, action):
        self.current_frame_index += 1
        if self.current_frame_index >= len(self.frames):
            self.current_frame_index = len(self.frames) - 1
        return self._get_obs(), 0, False, {}

    def seed(self, seed):
        pass

    def _get_obs(self):
        frame_path = self.frames[self.current_frame_index]
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return {'pov': frame, 'location_stats': {}, 'inventory': {}, 'damage_dealt': {'damage_dealt': np.array(0)}, 'kill_entity': {'villager': np.array(0)}}

    def __del__(self):
        import shutil
        if self.frames_dir:
            shutil.rmtree(self.frames_dir)


def _parse_bool(value: str, key: str) -> bool:
    normalized = value.strip().lower()
    truthy = {'1', 'true', 't', 'yes', 'y', 'on'}
    falsy = {'0', 'false', 'f', 'no', 'n', 'off'}
    if normalized in truthy:
        return True
    if normalized in falsy:
        return False
    raise ValueError(f"Invalid boolean for '{key}': '{value}'")


def _parse_placement_spec(spec: str) -> Dict[str, Any]:
    allowed_keys = {
        'side',
        'x',
        'y',
        'z',
        'tree_type',
        'relative',
        'villager',
        'villager_profession',
        'villager_type',
        'logs_above_villager',
    }
    if not spec or not spec.strip():
        raise ValueError("Empty --placement spec is not allowed")

    raw: Dict[str, str] = {}
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '=' not in chunk:
            raise ValueError(
                f"Invalid --placement chunk '{chunk}'. Expected key=value pairs separated by commas."
            )
        key, value = chunk.split('=', 1)
        key = key.strip().lower()
        value = value.strip()
        if key not in allowed_keys:
            raise ValueError(f"Unknown --placement key '{key}' in '{spec}'")
        if key in raw:
            raise ValueError(f"Duplicate --placement key '{key}' in '{spec}'")
        raw[key] = value

    placement: Dict[str, Any] = dict(PLACEMENT_DEFAULTS)

    if 'side' in raw:
        side = raw['side'].strip().lower()
        if side not in SIDE_DEFAULT_COORDS:
            allowed = ', '.join(sorted(SIDE_DEFAULT_COORDS.keys()))
            raise ValueError(f"Invalid side '{raw['side']}'. Expected one of: {allowed}")
        sx, sy, sz = SIDE_DEFAULT_COORDS[side]
        placement['x'] = sx
        placement['y'] = sy
        placement['z'] = sz

    if 'x' in raw:
        placement['x'] = int(raw['x'])
    if 'y' in raw:
        placement['y'] = int(raw['y'])
    if 'z' in raw:
        placement['z'] = int(raw['z'])
    if 'tree_type' in raw:
        placement['tree_type'] = raw['tree_type']
    if 'relative' in raw:
        placement['relative'] = _parse_bool(raw['relative'], 'relative')
    if 'villager' in raw:
        placement['villager'] = _parse_bool(raw['villager'], 'villager')
    if 'villager_profession' in raw:
        placement['villager_profession'] = raw['villager_profession']
    if 'villager_type' in raw:
        placement['villager_type'] = raw['villager_type']
    if 'logs_above_villager' in raw:
        placement['logs_above_villager'] = _parse_bool(raw['logs_above_villager'], 'logs_above_villager')
    return placement


def _build_creative_commands_from_args(args) -> list[str]:
    commands: list[str] = []
    supports_logs_above_villager = 'logs_above_villager' in inspect.signature(place_tree_commands).parameters
    for spec in args.placement:
        placement = _parse_placement_spec(spec)
        kwargs = {
            'tree_type': placement['tree_type'],
            'relative': placement['relative'],
            'villager': placement['villager'],
            'villager_profession': placement['villager_profession'],
            'villager_type': placement['villager_type'],
        }
        if supports_logs_above_villager:
            kwargs['logs_above_villager'] = placement['logs_above_villager']
        commands += place_tree_commands(
            placement['x'],
            placement['y'],
            placement['z'],
            **kwargs,
        )
    return commands


def _scalar(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _tp_command_from_pose(x: float, y: float, z: float, yaw: float | None = None, pitch: float | None = None) -> str:
    if yaw is None or pitch is None:
        return f"/tp @p {x:.3f} {y:.3f} {z:.3f}"
    return f"/tp @p {x:.3f} {y:.3f} {z:.3f} {yaw:.3f} {pitch:.3f}"


def _clone_action(action_template: dict) -> dict:
    return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in action_template.items()}


def _run_chat_command_step(env, action_template: dict, command: str):
    cmd_action = _clone_action(action_template)
    cmd_action["chat"] = command
    return env.step(cmd_action)


def _run_idle_steps(env, action_template: dict, count: int):
    if count <= 0:
        return None
    idle_action = _clone_action(action_template)
    idle_action["chat"] = ""
    obs = None
    for _ in range(count):
        obs, _, _, _ = env.step(idle_action)
    return obs


def _get_optional_yaw_pitch(obs) -> tuple[float, float] | None:
    location_stats = obs.get("location_stats", {})
    if "yaw" not in location_stats or "pitch" not in location_stats:
        return None
    return (_scalar(location_stats["yaw"]), _scalar(location_stats["pitch"]))


def _configure_command_feedback(env, action_template: dict, mode: str):
    setup_commands = []
    if mode == "gamerule_off":
        setup_commands = ["/gamerule sendCommandFeedback false"]
    elif mode == "gamerule_strict":
        setup_commands = [
            "/gamerule sendCommandFeedback false",
            "/gamerule logAdminCommands false",
        ]
    obs = None
    for command in setup_commands:
        obs, _, _, _ = _run_chat_command_step(env, action_template, command)
    return obs


def _suppress_attack(minerl_action: dict):
    if "attack" not in minerl_action:
        return
    attack_val = minerl_action["attack"]
    if isinstance(attack_val, np.ndarray):
        minerl_action["attack"] = np.zeros_like(attack_val)
    else:
        minerl_action["attack"] = 0


def run_agent(prompt_name, prompt_embed, gameplay_length, base_save_dirpath,
              in_model, in_weights, seed, cond_scale, save_additional_data, fake_env_zip,
              wipe_hidden_state_each_step, creative_commands, command_text,
              teleport_to_spawn_every, creative_command_wait_steps, teleport_wait_steps,
              teleport_feedback_mode, suppress_attack_actions, dummy_reset_before_seed):

    if fake_env_zip:
        env = FakeEnvironment(fake_env_zip)
        mineclip = load_mineclip_wconfig()
        agent = make_agent(in_model, in_weights, cond_scale=cond_scale)
    else:
        # Keep seed handling local so we can enforce deterministic reset ordering.
        agent, mineclip, env = load_mineclip_agent_env(in_model, in_weights, None, cond_scale)

    # Seed/reset ordering:
    # - optional dummy reset before first seeded rollout in a process
    # - set seed
    # - reset to apply the seed to world generation
    if seed is not None:
        if dummy_reset_before_seed:
            _ = env.reset()
        env.seed(seed)
        obs = env.reset()
    else:
        obs = env.reset()

    # Setup
    gameplay_frames = []
    # prog_evaluator = ProgrammaticEvaluator(obs)

    teleport_events = []
    teleport_enabled = False
    spawn_coords = None
    spawn_yaw_pitch = None
    spawn_keys = ("xpos", "ypos", "zpos")
    if teleport_to_spawn_every > 0:
        if all(k in obs["location_stats"] for k in spawn_keys):
            spawn_coords = tuple(_scalar(obs["location_stats"][k]) for k in spawn_keys)
            teleport_enabled = True
            spawn_yaw_pitch = _get_optional_yaw_pitch(obs)
            if spawn_yaw_pitch is None:
                print("Warning: spawn yaw/pitch missing; teleport will reset position but not camera view")
        else:
            print("Teleport-to-spawn disabled: location_stats missing xpos/ypos/zpos")

    commands = list(creative_commands)
    noop = {"buttons": np.array([[0]]), "camera": np.array([[0]])}
    base_action = agent._agent_action_to_env(noop)
    base_action["chat"] = ""

    if (commands or teleport_enabled) and teleport_feedback_mode != "default" and not fake_env_zip:
        try:
            setup_obs = _configure_command_feedback(env, base_action, teleport_feedback_mode)
            if setup_obs is not None:
                obs = setup_obs
        except Exception as exc:
            print(f"Warning: failed to configure command feedback mode '{teleport_feedback_mode}': {exc}")

    for command in commands:
        obs, _, _, _ = _run_chat_command_step(env, base_action, command)

    if commands and creative_command_wait_steps > 0:
        waited_obs = _run_idle_steps(env, base_action, creative_command_wait_steps)
        if waited_obs is not None:
            obs = waited_obs

    # Run agent in MineRL env
    kill_step = -1
    hits = []
    actions = []
    action_keys = sorted([k for k in base_action.keys() if k not in ['camera', 'chat']])
    location_stats = []
    location_keys = sorted(obs['location_stats'].keys())
    location_stats.append([obs['location_stats'][key].item() for key in location_keys])
    prev_inventory = dict(obs['inventory'])
    inventory_updates = []
    prev_dmg = 0
    pd_list = []
    first_frame_fullres = None
    globals.save_additional_data = save_additional_data
    for step in tqdm(range(gameplay_length)):
        did_teleport_this_step = False
        if (
            teleport_enabled
            and spawn_coords is not None
            and step > 0
            and (step % teleport_to_spawn_every == 0)
        ):
            pre_tp_xyz = tuple(_scalar(obs["location_stats"][k]) for k in spawn_keys)
            pre_tp_yaw_pitch = _get_optional_yaw_pitch(obs)
            if spawn_yaw_pitch is not None:
                teleport_cmd = _tp_command_from_pose(
                    spawn_coords[0], spawn_coords[1], spawn_coords[2],
                    yaw=spawn_yaw_pitch[0], pitch=spawn_yaw_pitch[1],
                )
            else:
                teleport_cmd = _tp_command_from_pose(*spawn_coords)
            obs, _, _, _ = _run_chat_command_step(env, base_action, teleport_cmd)
            if teleport_wait_steps > 0:
                waited_obs = _run_idle_steps(env, base_action, teleport_wait_steps)
                if waited_obs is not None:
                    obs = waited_obs
            post_tp_xyz = tuple(_scalar(obs["location_stats"][k]) for k in spawn_keys)
            post_tp_yaw_pitch = _get_optional_yaw_pitch(obs)
            pre_yaw = float("nan") if pre_tp_yaw_pitch is None else pre_tp_yaw_pitch[0]
            pre_pitch = float("nan") if pre_tp_yaw_pitch is None else pre_tp_yaw_pitch[1]
            post_yaw = float("nan") if post_tp_yaw_pitch is None else post_tp_yaw_pitch[0]
            post_pitch = float("nan") if post_tp_yaw_pitch is None else post_tp_yaw_pitch[1]
            teleport_events.append([
                int(step),
                pre_tp_xyz[0], pre_tp_xyz[1], pre_tp_xyz[2],
                post_tp_xyz[0], post_tp_xyz[1], post_tp_xyz[2],
                pre_yaw, pre_pitch,
                post_yaw, post_pitch,
                teleport_cmd,
                int(teleport_wait_steps),
            ])
            agent.reset(cond_scale=cond_scale)
            did_teleport_this_step = True

        if wipe_hidden_state_each_step and not did_teleport_this_step:
            agent.reset(cond_scale=cond_scale)
        with torch.cuda.amp.autocast():
            minerl_action, pd = agent.get_action_with_pd(obs, prompt_embed)

        if suppress_attack_actions:
            _suppress_attack(minerl_action)

        frame_fullres = obs['pov']
        if first_frame_fullres is None:
            first_frame_fullres = frame_fullres.copy()
        frame = cv2.resize(frame_fullres, (128, 128), interpolation=cv2.INTER_LINEAR)
        gameplay_frames.append(frame)
        obs, _, done, info = env.step(minerl_action)
        location_stats.append([obs['location_stats'][key].item() for key in location_keys])
        actions.append([minerl_action[key].item() for key in action_keys] + minerl_action['camera'][0].tolist())
        pd_list.append(pd)

        if done:
            if isinstance(info, dict) and info.get('error') is not None:
                print(f"Environment terminated early at step {step} with info['error']={info.get('error')}")
            else:
                print(f"Environment terminated early at step {step} (done=True)")
            break

        # prog_evaluator.update(obs)

        dmg = obs['damage_dealt']['damage_dealt'].item()
        if dmg > prev_dmg:
            hits.append([step, dmg - prev_dmg])
            prev_dmg = dmg
        if obs['kill_entity']['villager'] > 0 and kill_step == -1:
            kill_step = step
        if step > kill_step + 60 and kill_step != -1:  # finish episode 3 sec after villager kill
            break
        for key in obs['inventory'].keys():
            if obs['inventory'][key] != prev_inventory[key]:
                inventory_updates.append([step, key, obs['inventory'][key]])
                prev_inventory[key] = obs['inventory'][key]

    # Make the eval episode dir and save it and the logs
    dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dirpath = os.path.join(base_save_dirpath, f'{dt}')
    os.makedirs(run_dirpath, exist_ok=True)

    with open(os.path.join(run_dirpath, 'command.log'), mode='w') as command_file:
        command_file.write(command_text.strip() + "\n")

    if first_frame_fullres is not None:
        cv2.imwrite(
            os.path.join(run_dirpath, 'first_frame_fullres.png'),
            cv2.cvtColor(first_frame_fullres, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, 0],
        )

    save_frames_as_video(gameplay_frames, os.path.join(run_dirpath, 'video.mp4'), FPS, to_bgr=True)
    
    if save_additional_data:
        # save lossless observations:
        zip_filepath = os.path.join(run_dirpath, 'observations.zip')
        with zipfile.ZipFile(zip_filepath, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
            for idx, frame in enumerate(gameplay_frames):
                image_filepath = f'frame_{idx:06d}.png'
                cv2.imwrite(image_filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])
                zip_file.write(image_filepath, os.path.basename(image_filepath))
                os.remove(image_filepath)

        # save action probabilities:
        buttons = np.stack([pd['buttons'].cpu().numpy().squeeze() for pd in pd_list])
        camera = np.stack([pd['camera'].cpu().numpy().squeeze() for pd in pd_list])
        np.save(os.path.join(run_dirpath, 'buttons.npy'), buttons, allow_pickle=True)
        np.save(os.path.join(run_dirpath, 'camera.npy'), camera, allow_pickle=True)

        # save attention patterns:
        attn_patterns = np.stack(globals.attn_data)
        np.save(os.path.join(run_dirpath, 'attn_patterns.npy'), attn_patterns, allow_pickle=True)

    if teleport_events:
        with open(os.path.join(run_dirpath, 'teleport_events.log'), mode='w', newline='') as teleport_file:
            teleport_writer = csv.writer(teleport_file)
            teleport_writer.writerow([
                'step',
                'pre_x', 'pre_y', 'pre_z',
                'post_x', 'post_y', 'post_z',
                'pre_yaw', 'pre_pitch',
                'post_yaw', 'post_pitch',
                'command',
                'wait_steps',
            ])
            teleport_writer.writerows(teleport_events)

    with open(os.path.join(run_dirpath, 'hits.log'), mode='w', newline='') as hits_file:
        hits_writer = csv.writer(hits_file)
        hits_writer.writerow(['Step', 'Damage Dealt'])
        hits_writer.writerows(hits)

    with open(os.path.join(run_dirpath, 'kill_step.log'), mode='w') as kill_step_file:
        kill_step_file.write(str(kill_step))

    with open(os.path.join(run_dirpath, 'seed.log'), mode='w') as seed_file:
        seed_file.write(str(seed))

    with open(os.path.join(run_dirpath, 'prompt.log'), mode='w') as prompt_file:
        prompt_file.write(prompt_name)

    with open(os.path.join(run_dirpath, 'actions.log'), mode='w', newline='') as actions_file:
        actions_writer = csv.writer(actions_file)
        actions_writer.writerow(action_keys + ['camera1', 'camera2'])
        actions_writer.writerows(actions)

    with open(os.path.join(run_dirpath, 'location_stats.log'), mode='w', newline='') as location_stats_file:
        location_stats_writer = csv.writer(location_stats_file)
        location_stats_writer.writerow(location_keys)
        location_stats_writer.writerows(location_stats)

    with open(os.path.join(run_dirpath, 'inventory_updates.log'), mode='w', newline='') as inventory_updates_file:
        inventory_updates_writer = csv.writer(inventory_updates_file)
        inventory_updates_writer.writerow(['Step', 'Item', 'Count'])
        inventory_updates_writer.writerows(inventory_updates)

    # Print the programmatic eval task results at the end of the gameplay
    # prog_evaluator.print_results()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_model', type=str, default='data/weights/vpt/2x.model')
    parser.add_argument('--in_weights', type=str, default='data/weights/steve1/steve1.weights')
    parser.add_argument('--prior_weights', type=str, default='data/weights/steve1/steve1_prior.pt')
    parser.add_argument('--text_cond_scale', type=float, default=6.0)
    parser.add_argument('--visual_cond_scale', type=float, default=7.0)
    parser.add_argument('--gameplay_length', type=int, default=1000)
    parser.add_argument('--save_dirpath', type=str, default='data/generated_videos/')
    parser.add_argument('--custom_text_prompt', type=str, default=None)
    parser.add_argument('--custom_text_seed', type=int, default=None)
    parser.add_argument('--visual_prompt', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_additional_data', action='store_true', default=False, help='Save additional data if set')
    parser.add_argument('--fake_env_zip', type=str, default=None, help='Path to zip file containing fake environment frames')
    parser.add_argument('--wipe_hidden_state_each_step', action='store_true', default=False,
                        help='Reset agent hidden state every step to make actions depend only on current observation')
    parser.add_argument('--teleport_to_spawn_every', type=int, default=100,
                        help='Run "/tp @p <spawn>" every N policy steps. Use <=0 to disable.')
    parser.add_argument('--creative_command_wait_steps', type=int, default=10,
                        help='Idle steps after initial creative commands.')
    parser.add_argument('--teleport_wait_steps', type=int, default=10,
                        help='Idle steps after each teleport command.')
    parser.add_argument('--teleport_feedback_mode', type=str, default='gamerule_strict',
                        choices=['default', 'gamerule_off', 'gamerule_strict'],
                        help='How to suppress command feedback text for creative/teleport commands.')
    parser.add_argument('--suppress_attack_actions', action='store_true', default=False,
                        help='Force attack action to 0 on every policy step.')
    parser.add_argument('--dummy_reset_before_seed', action='store_true', default=False,
                        help='Do one extra reset before applying seed, then reset again.')
    parser.add_argument('--placement', action='append', default=[],
                        help=('Repeatable placement spec. '
                              'Format: key=value[,key=value...]. '
                              'Supported keys: side(left/right), x, y, z, tree_type, relative, villager, '
                              'villager_profession, villager_type, logs_above_villager. '
                              'Example: --placement side=left,villager=true,logs_above_villager=true'))

    args = parser.parse_args()
    command_text = " ".join(shlex.quote(arg) for arg in [sys.executable, *sys.argv])
    try:
        creative_commands = _build_creative_commands_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    has_custom_text_prompt = args.custom_text_prompt is not None
    has_visual_prompt = args.visual_prompt is not None
    if has_custom_text_prompt == has_visual_prompt:
        parser.error('Specify exactly one of --custom_text_prompt or --visual_prompt')

    if has_custom_text_prompt:
        print(f"\nGenerating video for prompt: {args.custom_text_prompt}")
        mineclip = load_mineclip_wconfig()
        prior = load_vae_model(PRIOR_INFO)
        prompt_embed = get_prior_embed(args.custom_text_prompt, mineclip, prior, DEVICE, seed=args.custom_text_seed)
        run_agent(args.custom_text_prompt, prompt_embed, args.gameplay_length, args.save_dirpath, args.in_model,
                  args.in_weights, args.seed, args.text_cond_scale, args.save_additional_data, args.fake_env_zip,
                  args.wipe_hidden_state_each_step, creative_commands, command_text,
                  args.teleport_to_spawn_every, args.creative_command_wait_steps, args.teleport_wait_steps,
                  args.teleport_feedback_mode, args.suppress_attack_actions, args.dummy_reset_before_seed)
    else:
        visual_prompt_embeds = load_visual_prompt_embeds()
        if args.visual_prompt not in visual_prompt_embeds:
            parser.error(f"Unknown --visual_prompt '{args.visual_prompt}'")
        print(f"\nGenerating video for prompt: {args.visual_prompt}")
        run_agent(args.visual_prompt, visual_prompt_embeds[args.visual_prompt], args.gameplay_length, args.save_dirpath,
                  args.in_model, args.in_weights, args.seed, args.visual_cond_scale, args.save_additional_data,
                  args.fake_env_zip, args.wipe_hidden_state_each_step, creative_commands, command_text,
                  args.teleport_to_spawn_every, args.creative_command_wait_steps, args.teleport_wait_steps,
                  args.teleport_feedback_mode, args.suppress_attack_actions, args.dummy_reset_before_seed)

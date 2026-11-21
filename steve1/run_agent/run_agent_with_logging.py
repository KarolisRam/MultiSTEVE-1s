import argparse
import csv
import os
import shlex
import shutil
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
}

SIDE_DEFAULT_COORDS = {
    'left': (4, 0, 7),
    'right': (-4, 0, 7),
}

FAKE_ENV_SOURCE_FILENAMES = (
    'video.mp4',
    'hits.log',
    'kill_step.log',
    'actions.log',
    'location_stats.log',
    'inventory_updates.log',
)


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


def _copy_fake_env_source_artifacts(fake_env_zip: str, run_dirpath: str, save_additional_data: bool) -> None:
    source_episode_dir = os.path.dirname(os.path.abspath(fake_env_zip))
    filenames = list(FAKE_ENV_SOURCE_FILENAMES)
    if save_additional_data:
        filenames.append('observations.zip')
    for filename in filenames:
        source_path = os.path.join(source_episode_dir, filename)
        if os.path.isfile(source_path):
            shutil.copyfile(source_path, os.path.join(run_dirpath, filename))


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
    return placement


def _build_creative_commands_from_args(args) -> list[str]:
    commands: list[str] = []
    for spec in args.placement:
        placement = _parse_placement_spec(spec)
        commands += place_tree_commands(
            placement['x'],
            placement['y'],
            placement['z'],
            tree_type=placement['tree_type'],
            relative=placement['relative'],
            villager=placement['villager'],
            villager_profession=placement['villager_profession'],
            villager_type=placement['villager_type'],
        )
    return commands


def run_agent(prompt_name, prompt_embed, gameplay_length, base_save_dirpath,
              in_model, in_weights, seed, cond_scale, save_additional_data, fake_env_zip,
              wipe_hidden_state_each_step, creative_commands, command_text):

    if fake_env_zip:
        env = FakeEnvironment(fake_env_zip)
        mineclip = load_mineclip_wconfig()
        agent = make_agent(in_model, in_weights, cond_scale=cond_scale)
    else:
        agent, mineclip, env = load_mineclip_agent_env(in_model, in_weights, seed, cond_scale)

    # Make sure seed is set if specified
    obs = env.reset()
    if seed is not None:
        env.seed(seed)

    # Setup
    gameplay_frames = []
    # prog_evaluator = ProgrammaticEvaluator(obs)

    # Creative stuff
    commands = list(creative_commands)
    noop = {"buttons": np.array([[0]]), "camera": np.array([[0]])}
    minerl_action = agent._agent_action_to_env(noop)
    for command in commands:
        minerl_action["chat"] = command
        env.step(minerl_action)
        # env.render()

    minerl_action["chat"] = ""
    if len(commands) > 0:
        for _ in range(250):  # wait for chat to disappear.
            # for example maybe add “[distance=..12]” to the commands, does that work?
            # Can you suppress chat output from a creative command? Can you disable chat?
            obs, reward, done, info = env.step(minerl_action)
            # env.render()

    # Run agent in MineRL env
    kill_step = -1
    hits = []
    actions = []
    action_keys = sorted([k for k in minerl_action.keys() if k not in ['camera', 'chat']])
    location_stats = []
    location_keys = sorted(obs['location_stats'].keys())
    location_stats.append([obs['location_stats'][key].item() for key in location_keys])
    prev_inventory = dict(obs['inventory'])
    inventory_updates = []
    prev_dmg = 0
    pd_list = []
    globals.save_additional_data = save_additional_data
    for step in tqdm(range(gameplay_length)):
        if wipe_hidden_state_each_step:
            agent.reset(cond_scale=cond_scale)
        with torch.cuda.amp.autocast():
            minerl_action, pd = agent.get_action_with_pd(obs, prompt_embed)

        frame = obs['pov']
        frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LINEAR)
        gameplay_frames.append(frame)
        obs, _, _, _ = env.step(minerl_action)
        location_stats.append([obs['location_stats'][key].item() for key in location_keys])
        actions.append([minerl_action[key].item() for key in action_keys] + minerl_action['camera'][0].tolist())
        pd_list.append(pd)

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

    if fake_env_zip:
        _copy_fake_env_source_artifacts(fake_env_zip, run_dirpath, save_additional_data)

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
    parser.add_argument('--placement', action='append', default=[],
                        help=('Repeatable placement spec. '
                              'Format: key=value[,key=value...]. '
                              'Supported keys: side(left/right), x, y, z, tree_type, relative, villager, '
                              'villager_profession, villager_type. '
                              'Example: --placement side=left,villager=true'))

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
                  args.wipe_hidden_state_each_step, creative_commands, command_text)
    else:
        visual_prompt_embeds = load_visual_prompt_embeds()
        if args.visual_prompt not in visual_prompt_embeds:
            parser.error(f"Unknown --visual_prompt '{args.visual_prompt}'")
        print(f"\nGenerating video for prompt: {args.visual_prompt}")
        run_agent(args.visual_prompt, visual_prompt_embeds[args.visual_prompt], args.gameplay_length, args.save_dirpath,
                  args.in_model, args.in_weights, args.seed, args.visual_cond_scale, args.save_additional_data,
                  args.fake_env_zip, args.wipe_hidden_state_each_step, creative_commands, command_text)

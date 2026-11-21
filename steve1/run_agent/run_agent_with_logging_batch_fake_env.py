import argparse
import csv
import os
import shlex
import sys
import zipfile
from datetime import datetime
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

import globals
from steve1.config import PRIOR_INFO, DEVICE
from steve1.data.text_alignment.vae import load_vae_model
from steve1.run_agent.paper_prompts import load_visual_prompt_embeds
from steve1.run_agent.programmatic_eval import ProgrammaticEvaluator
from steve1.utils.embed_utils import get_prior_embed
from steve1.utils.mineclip_agent_env_utils import load_mineclip_agent_env, load_mineclip_wconfig, make_agent
from steve1.utils.video_utils import save_frames_as_video
from steve1.VPT.agent import resize_image, AGENT_RESOLUTION
from steve1.VPT.lib.tree_util import tree_map, tree_multimap

FPS = 20
TIMINGS = {}
TIMING_ORDER = []


def _add_timing(label, seconds):
    if label not in TIMING_ORDER:
        TIMING_ORDER.append(label)
    TIMINGS[label] = TIMINGS.get(label, 0.0) + seconds


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


def _repeat_embed_for_batch(prompt_embed, batch_size, device):
    embed = torch.from_numpy(prompt_embed).to(device)
    if embed.ndim == 1:
        embed = embed.unsqueeze(0)
    if embed.shape[0] == batch_size:
        return embed
    if embed.shape[0] != 1:
        raise ValueError(f"prompt_embed batch dim {embed.shape[0]} does not match batch size {batch_size}")
    repeat_shape = [batch_size] + [1] * (embed.ndim - 1)
    return embed.repeat(*repeat_shape)


def _repeat_embed_for_sequence(prompt_embed, batch_size, timesteps, device):
    embed = torch.from_numpy(prompt_embed).to(device)
    if embed.ndim == 1:
        embed = embed.unsqueeze(0)
    if embed.shape[0] not in (1, batch_size):
        raise ValueError(f"prompt_embed batch dim {embed.shape[0]} does not match batch size {batch_size}")
    if embed.shape[0] == 1:
        embed = embed.repeat(batch_size, 1)
    embed = embed.unsqueeze(1).repeat(1, timesteps, 1)
    return embed


def _build_agent_input(frames, prompt_embed, device):
    resized = np.stack([resize_image(frame, AGENT_RESOLUTION) for frame in frames], axis=0)
    agent_input = {"img": torch.from_numpy(resized).to(device)}
    agent_input["mineclip_embed"] = _repeat_embed_for_batch(prompt_embed, len(frames), device)
    return agent_input


def _collect_pd_batch(agent, frames, prompt_embed, cond_scale):
    device = agent.device
    batch_size = len(frames)
    agent_input = _build_agent_input(frames, prompt_embed, device)
    first = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    state_in = agent.policy.initial_state(batch_size)

    with torch.cuda.amp.autocast():
        if cond_scale is None:
            _, _, result = agent.policy.act(
                agent_input, first, state_in, stochastic=True, return_pd=True
            )
            pd = result["pd"]
        else:
            _, _, result_cond = agent.policy.act(
                agent_input, first, state_in, stochastic=True, return_pd=True
            )
            uncond_input = dict(agent_input)
            uncond_input["mineclip_embed"] = torch.zeros_like(agent_input["mineclip_embed"])
            _, _, result_uncond = agent.policy.act(
                uncond_input, first, state_in, stochastic=True, return_pd=True
            )
            pd = tree_multimap(
                lambda x, y: ((1 + cond_scale) * x) - (cond_scale * y),
                result_cond["pd"],
                result_uncond["pd"],
            )
    return pd


def _collect_pd_sequence(agent, frames, prompt_embed, cond_scale, state_in):
    device = agent.device
    timesteps = len(frames)
    resized = np.stack([resize_image(frame, AGENT_RESOLUTION) for frame in frames], axis=0)
    img = torch.from_numpy(resized).to(device).unsqueeze(0)
    with torch.no_grad(), torch.cuda.amp.autocast():
        if cond_scale is None:
            obs = {
                "img": img,
                "mineclip_embed": _repeat_embed_for_sequence(prompt_embed, 1, timesteps, device),
            }
            first = torch.zeros((1, timesteps), dtype=torch.bool, device=device)
            (pd, _, _), state_out = agent.policy(obs=obs, first=first, state_in=state_in)
            pd = tree_map(lambda x: x[0], pd)
        else:
            img = img.repeat(2, 1, 1, 1, 1)
            embed_cond = _repeat_embed_for_sequence(prompt_embed, 1, timesteps, device)
            obs = {
                "img": img,
                "mineclip_embed": torch.cat([embed_cond, torch.zeros_like(embed_cond)], dim=0),
            }
            first = torch.zeros((2, timesteps), dtype=torch.bool, device=device)
            (pd, _, _), state_out = agent.policy(obs=obs, first=first, state_in=state_in)
            pd = tree_map(lambda x: ((1 + cond_scale) * x[0]) - (cond_scale * x[1]), pd)
    return pd, state_out


def run_agent(prompt_name, prompt_embed, gameplay_length, base_save_dirpath,
              in_model, in_weights, seed, cond_scale, save_additional_data, fake_env_zip,
              wipe_hidden_state_each_step, batch_size, command_text):

    if fake_env_zip:
        env = FakeEnvironment(fake_env_zip)
        start_time = time.time()
        # mineclip = load_mineclip_wconfig()
        _add_timing("load_mineclip_wconfig", time.time() - start_time)
        start_time = time.time()
        agent = make_agent(in_model, in_weights, cond_scale=cond_scale)
        _add_timing("load_agent", time.time() - start_time)
    else:
        start_time = time.time()
        agent, mineclip, env = load_mineclip_agent_env(in_model, in_weights, seed, cond_scale)
        _add_timing("load_agent_env", time.time() - start_time)

    # Make sure seed is set if specified
    start_time = time.time()
    obs = env.reset()
    _add_timing("env_reset", time.time() - start_time)
    if seed is not None:
        env.seed(seed)

    # Setup
    gameplay_frames = []
    # prog_evaluator = ProgrammaticEvaluator(obs)

    commands = []
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
    if fake_env_zip:
        total_frames = min(gameplay_length, len(env.frames))
        pd_buttons_list = []
        pd_camera_list = []
        effective_batch_size = max(1, batch_size)
        eval_start_time = time.time()
        if wipe_hidden_state_each_step:
            for start_idx in tqdm(range(0, total_frames, effective_batch_size)):
                batch_paths = env.frames[start_idx:start_idx + effective_batch_size]
                frames = []
                for frame_path in batch_paths:
                    frame = cv2.imread(frame_path)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                pd = _collect_pd_batch(agent, frames, prompt_embed, cond_scale)
                pd_buttons = pd["buttons"].detach().cpu().numpy()
                pd_camera = pd["camera"].detach().cpu().numpy()
                for i in range(len(frames)):
                    pd_buttons_list.append(np.squeeze(pd_buttons[i]))
                    pd_camera_list.append(np.squeeze(pd_camera[i]))
        else:
            if cond_scale is None:
                state = agent.policy.initial_state(1)
            else:
                state = agent.policy.initial_state(2)
            for start_idx in tqdm(range(0, total_frames, effective_batch_size)):
                batch_paths = env.frames[start_idx:start_idx + effective_batch_size]
                frames = []
                for frame_path in batch_paths:
                    frame = cv2.imread(frame_path)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                pd, state = _collect_pd_sequence(agent, frames, prompt_embed, cond_scale, state)
                pd_buttons = pd["buttons"].detach().cpu().numpy()
                pd_camera = pd["camera"].detach().cpu().numpy()
                for i in range(len(frames)):
                    pd_buttons_list.append(np.squeeze(pd_buttons[i]))
                    pd_camera_list.append(np.squeeze(pd_camera[i]))
        _add_timing("batch_eval", time.time() - eval_start_time)
    else:
        eval_start_time = time.time()
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
        _add_timing("step_eval", time.time() - eval_start_time)

    # Make the eval episode dir and save it and the logs
    dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dirpath = os.path.join(base_save_dirpath, f'{dt}')
    os.makedirs(run_dirpath, exist_ok=True)

    with open(os.path.join(run_dirpath, 'command.log'), mode='w') as command_file:
        command_file.write(command_text.strip() + "\n")
    
    if not fake_env_zip:
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

    if fake_env_zip:
        buttons = np.stack(pd_buttons_list)
        camera = np.stack(pd_camera_list)
        np.save(os.path.join(run_dirpath, 'buttons.npy'), buttons, allow_pickle=True)
        np.save(os.path.join(run_dirpath, 'camera.npy'), camera, allow_pickle=True)
    else:
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
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for fake-env evaluation')

    args = parser.parse_args()
    command_text = " ".join(shlex.quote(arg) for arg in [sys.executable, *sys.argv])

    has_custom_text_prompt = args.custom_text_prompt is not None
    has_visual_prompt = args.visual_prompt is not None
    if has_custom_text_prompt == has_visual_prompt:
        parser.error('Specify exactly one of --custom_text_prompt or --visual_prompt')

    if has_custom_text_prompt:
        start_time = time.time()
        print(f"\nGenerating video for prompt: {args.custom_text_prompt}")
        mineclip = load_mineclip_wconfig()
        _add_timing("load_mineclip_wconfig", time.time() - start_time)
        start_time = time.time()
        prior = load_vae_model(PRIOR_INFO)
        _add_timing("load_prior", time.time() - start_time)
        prompt_embed = get_prior_embed(args.custom_text_prompt, mineclip, prior, DEVICE, seed=args.custom_text_seed)
        run_agent(args.custom_text_prompt, prompt_embed, args.gameplay_length, args.save_dirpath, args.in_model,
                  args.in_weights, args.seed, args.text_cond_scale, args.save_additional_data, args.fake_env_zip,
                  args.wipe_hidden_state_each_step, args.batch_size, command_text)
    else:
        start_time = time.time()
        visual_prompt_embeds = load_visual_prompt_embeds()
        _add_timing("load_visual_prompt_embeds", time.time() - start_time)
        if args.visual_prompt not in visual_prompt_embeds:
            parser.error(f"Unknown --visual_prompt '{args.visual_prompt}'")
        print(f"\nGenerating video for prompt: {args.visual_prompt}")
        run_agent(args.visual_prompt, visual_prompt_embeds[args.visual_prompt], args.gameplay_length, args.save_dirpath,
                  args.in_model, args.in_weights, args.seed, args.visual_cond_scale, args.save_additional_data,
                  args.fake_env_zip, args.wipe_hidden_state_each_step, args.batch_size, command_text)

    if TIMINGS:
        total_time = sum(TIMINGS.values())
        print("\nTiming summary:")
        print(f"{'Section':<28} {'Seconds':>10} {'Percent':>9}")
        for label in TIMING_ORDER:
            seconds = TIMINGS[label]
            percent = (seconds / total_time * 100.0) if total_time > 0 else 0.0
            print(f"{label:<28} {seconds:>10.2f} {percent:>8.1f}%")
        print(f"{'Total':<28} {total_time:>10.2f} {100.0:>8.1f}%")
    sys.exit(0)

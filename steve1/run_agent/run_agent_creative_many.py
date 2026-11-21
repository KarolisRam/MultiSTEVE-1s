import argparse
import csv
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import torch
from tqdm import tqdm

from multisteve.mc_creative import place_tree_commands
from steve1.config import PRIOR_INFO, DEVICE
from steve1.data.text_alignment.vae import load_vae_model
from steve1.run_agent.paper_prompts import load_text_prompt_embeds, load_visual_prompt_embeds
from steve1.run_agent.programmatic_eval import ProgrammaticEvaluator
from steve1.utils.embed_utils import get_prior_embed
from steve1.utils.mineclip_agent_env_utils import load_mineclip_agent_env, load_mineclip_wconfig
from steve1.utils.video_utils import save_frames_as_video

FPS = 20


def run_agent(prompt_embed, gameplay_length, save_video_filepath,
              in_model, in_weights, seed, cond_scale):
    assert cond_scale is not None
    agent, mineclip, env = load_mineclip_agent_env(in_model, in_weights, seed, cond_scale)

    # Make sure seed is set if specified
    obs = env.reset()
    seed = 49
    if seed is not None:
        env.seed(seed)

    # Setup
    gameplay_frames = []
    prog_evaluator = ProgrammaticEvaluator(obs)

    # Creative stuff
    commands = []
    # commands += place_tree_commands(4, 0, 7, villager=left_object == "villager")
    # commands += place_tree_commands(-4, 0, 7, villager=right_object == "villager")
    commands += place_tree_commands(4, 0, 7, villager=True)
    # commands += place_tree_commands(-4, 0, 7, villager=False)
    noop = {"buttons": np.array([[0]]), "camera": np.array([[0]])}
    minerl_action = agent._agent_action_to_env(noop)
    for command in commands:
        minerl_action["chat"] = command
        env.step(minerl_action)
        # env.render()

    minerl_action["chat"] = ""
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
    prev_dmg = 0
    for step in tqdm(range(gameplay_length)):
        with torch.cuda.amp.autocast():
            minerl_action = agent.get_action(obs, prompt_embed)

        obs, _, _, _ = env.step(minerl_action)
        frame = obs['pov']
        frame = cv2.resize(frame, (128, 128))
        gameplay_frames.append(frame)
        location_stats.append([obs['location_stats'][key].item() for key in location_keys])
        actions.append([minerl_action[key].item() for key in action_keys] + minerl_action['camera'][0].tolist())

        prog_evaluator.update(obs)

        dmg = obs['damage_dealt']['damage_dealt'].item()
        if dmg > prev_dmg:
            hits.append([step, dmg - prev_dmg])
            prev_dmg = dmg
        if obs['kill_entity']['villager'] > 0 and kill_step == -1:
            kill_step = step
        if step > kill_step + 60 and kill_step != -1:  # finish episode 3 sec after villager kill
            break

    # Make the eval episode dir and save it and the logs
    dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dirpath = os.path.join(os.path.dirname(save_video_filepath), f'{dt}')
    os.makedirs(save_dirpath, exist_ok=True)

    save_frames_as_video(gameplay_frames, os.path.join(save_dirpath, f'video.mp4'), FPS, to_bgr=True)

    with open(os.path.join(save_dirpath, 'hits.log'), mode='w', newline='') as hits_file:
        hits_writer = csv.writer(hits_file)
        hits_writer.writerow(['Step', 'Damage Dealt'])
        hits_writer.writerows(hits)

    # save kill_step in .txt:
    with open(os.path.join(save_dirpath, 'kill_step.log'), mode='w') as kill_step_file:
        kill_step_file.write(str(kill_step))

    # save filename of save_video_filepath in .txt:
    with open(os.path.join(save_dirpath, 'prompt.log'), mode='w') as save_video_filepath_file:
        save_video_filepath_file.write(os.path.splitext(os.path.basename(save_video_filepath))[0].split("'")[1])
        
    # save actions as csv:
    with open(os.path.join(save_dirpath, 'actions.log'), mode='w', newline='') as actions_file:
        actions_writer = csv.writer(actions_file)
        actions_writer.writerow(action_keys + ['camera1', 'camera2'])
        actions_writer.writerows(actions)

    # save location stats as csv:
    with open(os.path.join(save_dirpath, 'location_stats.log'), mode='w', newline='') as location_stats_file:
        location_stats_writer = csv.writer(location_stats_file)
        location_stats_writer.writerow(location_keys)
        location_stats_writer.writerows(location_stats)

    # Print the programmatic eval task results at the end of the gameplay
    prog_evaluator.print_results()


def generate_text_prompt_videos(prompt_embeds, in_model, in_weights, seed, cond_scale, gameplay_length, save_dirpath):
    for name, prompt_embed in prompt_embeds.items():
        print(f'\nGenerating video for text prompt with name: {name}')
        save_video_filepath = os.path.join(save_dirpath, f'\'{name}\' - Text Prompt.mp4')
        if not os.path.exists(save_video_filepath):
            run_agent(prompt_embed, gameplay_length, save_video_filepath,
                      in_model, in_weights, seed, cond_scale)
        else:
            print(f'Video already exists at {save_video_filepath}, skipping...')


def generate_visual_prompt_videos(prompt_embeds, in_model, in_weights, seed, cond_scale, gameplay_length, save_dirpath):
    for name, prompt_embed in prompt_embeds.items():
        print(f'\nGenerating video for visual prompt with name: {name}')
        save_video_filepath = os.path.join(save_dirpath, f'{name} - Visual Prompt.mp4')
        if not os.path.exists(save_video_filepath):
            run_agent(prompt_embed, gameplay_length, save_video_filepath,
                      in_model, in_weights, seed, cond_scale)
        else:
            print(f'Video already exists at {save_video_filepath}, skipping...')


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
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.custom_text_prompt is not None:
        # Generate a video for the text prompt
        mineclip = load_mineclip_wconfig()
        prior = load_vae_model(PRIOR_INFO)
        prompt_embed = get_prior_embed(args.custom_text_prompt, mineclip, prior, DEVICE)
        custom_prompt_embeds = {args.custom_text_prompt: prompt_embed}
        generate_text_prompt_videos(custom_prompt_embeds, args.in_model, args.in_weights, args.seed,
                                    args.text_cond_scale, args.gameplay_length, args.save_dirpath)
    else:
        # Generate videos for the text and visual prompts used in the paper
        text_prompt_embeds = load_text_prompt_embeds()
        visual_prompt_embeds = load_visual_prompt_embeds()
        generate_text_prompt_videos(text_prompt_embeds, args.in_model, args.in_weights, args.seed,
                                    args.text_cond_scale, args.gameplay_length, args.save_dirpath)
        generate_visual_prompt_videos(visual_prompt_embeds, args.in_model, args.in_weights, args.seed,
                                      args.visual_cond_scale, args.gameplay_length, args.save_dirpath)
        sys.exit(0)

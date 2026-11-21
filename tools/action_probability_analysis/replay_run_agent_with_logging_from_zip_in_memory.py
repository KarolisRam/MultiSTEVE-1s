import argparse
import shlex
import sys
import time
import zipfile

import cv2
import numpy as np

from steve1.config import PRIOR_INFO, DEVICE
from steve1.data.text_alignment.vae import load_vae_model
from steve1.run_agent import run_agent_with_logging as runner
from steve1.run_agent.paper_prompts import load_visual_prompt_embeds
from steve1.utils.embed_utils import get_prior_embed
from steve1.utils.mineclip_agent_env_utils import load_mineclip_wconfig


class InMemoryFakeEnvironment:
    def __init__(self, zip_filepath):
        self.zip_filepath = zip_filepath
        self.frames = []
        self.current_frame_index = 0

    def reset(self):
        self.current_frame_index = 0
        self.frames = []
        with zipfile.ZipFile(self.zip_filepath, "r") as zip_ref:
            frame_names = sorted(name for name in zip_ref.namelist() if name.endswith(".png"))
            for frame_name in frame_names:
                encoded = np.frombuffer(zip_ref.read(frame_name), dtype=np.uint8)
                frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(frame)
        return self._get_obs()

    def step(self, action):
        self.current_frame_index += 1
        if self.current_frame_index >= len(self.frames):
            self.current_frame_index = len(self.frames) - 1
        return self._get_obs(), 0, False, {}

    def seed(self, seed):
        pass

    def _get_obs(self):
        frame = self.frames[self.current_frame_index]
        return {
            "pov": frame,
            "location_stats": {},
            "inventory": {},
            "damage_dealt": {"damage_dealt": np.array(0)},
            "kill_entity": {"villager": np.array(0)},
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_model', type=str, default='data/weights/vpt/2x.model')
    parser.add_argument('--in_weights', type=str, default='data/weights/steve1/steve1.weights')
    parser.add_argument('--prior_weights', type=str, default='data/weights/steve1/steve1_prior.pt')
    parser.add_argument('--text_cond_scale', type=float, default=6.0)
    parser.add_argument('--visual_cond_scale', type=float, default=7.0)
    parser.add_argument('--gameplay_length', type=int, default=1000)
    parser.add_argument('--save_dirpath', type=str, required=True)
    parser.add_argument('--custom_text_prompt', type=str, default=None)
    parser.add_argument('--custom_text_seed', type=int, default=None)
    parser.add_argument('--visual_prompt', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_additional_data', action='store_true', default=False)
    parser.add_argument('--fake_env_zip', type=str, required=True)
    parser.add_argument('--wipe_hidden_state_each_step', action='store_true', default=False)
    args = parser.parse_args()

    command_text = " ".join(shlex.quote(arg) for arg in [sys.executable, *sys.argv])
    has_custom_text_prompt = args.custom_text_prompt is not None
    has_visual_prompt = args.visual_prompt is not None
    if has_custom_text_prompt == has_visual_prompt:
        parser.error('Specify exactly one of --custom_text_prompt or --visual_prompt')

    runner.FakeEnvironment = InMemoryFakeEnvironment

    start = time.perf_counter()
    if has_custom_text_prompt:
        print(f"\nReplaying observations in-memory for prompt: {args.custom_text_prompt}")
        mineclip = load_mineclip_wconfig()
        prior = load_vae_model(PRIOR_INFO)
        prompt_embed = get_prior_embed(args.custom_text_prompt, mineclip, prior, DEVICE, seed=args.custom_text_seed)
        runner.run_agent(args.custom_text_prompt, prompt_embed, args.gameplay_length, args.save_dirpath, args.in_model,
                         args.in_weights, args.seed, args.text_cond_scale, args.save_additional_data, args.fake_env_zip,
                         args.wipe_hidden_state_each_step, [], command_text)
    else:
        visual_prompt_embeds = load_visual_prompt_embeds()
        if args.visual_prompt not in visual_prompt_embeds:
            parser.error(f"Unknown --visual_prompt '{args.visual_prompt}'")
        print(f"\nReplaying observations in-memory for prompt: {args.visual_prompt}")
        runner.run_agent(args.visual_prompt, visual_prompt_embeds[args.visual_prompt], args.gameplay_length, args.save_dirpath,
                         args.in_model, args.in_weights, args.seed, args.visual_cond_scale, args.save_additional_data,
                         args.fake_env_zip, args.wipe_hidden_state_each_step, [], command_text)
    elapsed = time.perf_counter() - start
    print(f"\nIn-memory replay wall time: {elapsed:.4f} seconds")


if __name__ == '__main__':
    main()

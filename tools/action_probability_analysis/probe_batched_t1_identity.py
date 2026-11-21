import argparse
import json
import os
import time
import zipfile

import cv2
import numpy as np
import torch

from steve1.config import PRIOR_INFO, DEVICE
from steve1.data.text_alignment.vae import load_vae_model
from steve1.utils.embed_utils import get_prior_embed
from steve1.utils.mineclip_agent_env_utils import load_mineclip_wconfig, make_agent
from steve1.VPT.agent import AGENT_RESOLUTION, resize_image
from steve1.VPT.lib.tree_util import tree_map, tree_multimap


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


def _state_cat(state_a, state_b):
    if state_a is None and state_b is None:
        return None
    if isinstance(state_a, torch.Tensor):
        return torch.cat([state_a, state_b], dim=0)
    if isinstance(state_a, tuple):
        return tuple(_state_cat(a, b) for a, b in zip(state_a, state_b))
    if isinstance(state_a, list):
        return [_state_cat(a, b) for a, b in zip(state_a, state_b)]
    raise TypeError(f"Unsupported state type for concat: {type(state_a)}")


def _state_slice(state, start, end):
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state[start:end]
    if isinstance(state, tuple):
        return tuple(_state_slice(v, start, end) for v in state)
    if isinstance(state, list):
        return [_state_slice(v, start, end) for v in state]
    raise TypeError(f"Unsupported state type for slicing: {type(state)}")


def _policy_forward_pd(agent, agent_input, first, state_in):
    obs_with_t = tree_map(lambda x: x.unsqueeze(1), agent_input)
    first_with_t = first.unsqueeze(1)
    (pd, _, _), state_out = agent.policy(obs=obs_with_t, first=first_with_t, state_in=state_in)
    return tree_map(lambda x: x[:, 0], pd), state_out


def _decode_frames_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        frame_names = sorted(name for name in zf.namelist() if name.endswith(".png"))
        frames = []
        for name in frame_names:
            encoded = np.frombuffer(zf.read(name), dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    return frames


def _summarize_against_reference(reference, candidate):
    diff = candidate - reference
    abs_diff = np.abs(diff)
    return {
        "shape": list(reference.shape),
        "exact_equal": bool(np.array_equal(reference, candidate)),
        "mean_abs_diff": float(abs_diff.mean()),
        "max_abs_diff": float(abs_diff.max()),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_model", required=True)
    parser.add_argument("--in_weights", required=True)
    parser.add_argument("--zip_path", required=True)
    parser.add_argument("--reference_run_dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt_seed", type=int, required=True)
    parser.add_argument("--cond_scale", type=float, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mineclip = load_mineclip_wconfig()
    prior = load_vae_model(PRIOR_INFO)
    prompt_embed = get_prior_embed(args.prompt, mineclip, prior, DEVICE, seed=args.prompt_seed)
    agent = make_agent(args.in_model, args.in_weights, cond_scale=args.cond_scale)

    reference_buttons = np.load(os.path.join(args.reference_run_dir, "buttons.npy"), allow_pickle=True)
    reference_camera = np.load(os.path.join(args.reference_run_dir, "camera.npy"), allow_pickle=True)
    frames = _decode_frames_from_zip(args.zip_path)

    batch_size = args.batch_size
    device = agent.device
    state_cond = agent.policy.initial_state(batch_size)
    state_uncond = agent.policy.initial_state(batch_size)

    buttons_out = []
    camera_out = []
    intra_batch_identical_buttons = True
    intra_batch_identical_camera = True

    start = time.perf_counter()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for frame in frames:
            resized = resize_image(frame, AGENT_RESOLUTION)
            img = np.stack([resized] * batch_size, axis=0)
            agent_input = {"img": torch.from_numpy(img).to(device)}
            agent_input["mineclip_embed"] = _repeat_embed_for_batch(prompt_embed, batch_size, device)
            first = torch.zeros((batch_size,), dtype=torch.bool, device=device)

            merged_input = {
                "img": torch.cat([agent_input["img"], agent_input["img"]], dim=0),
                "mineclip_embed": torch.cat(
                    [agent_input["mineclip_embed"], torch.zeros_like(agent_input["mineclip_embed"])], dim=0
                ),
            }
            merged_first = torch.cat([first, first], dim=0)
            merged_state = _state_cat(state_cond, state_uncond)
            merged_pd, merged_next_state = _policy_forward_pd(agent, merged_input, merged_first, merged_state)

            pd_cond = tree_map(lambda x: x[:batch_size], merged_pd)
            pd_uncond = tree_map(lambda x: x[batch_size:], merged_pd)
            pd = tree_multimap(
                lambda x, y: ((1 + args.cond_scale) * x) - (args.cond_scale * y),
                pd_cond,
                pd_uncond,
            )
            state_cond = _state_slice(merged_next_state, 0, batch_size)
            state_uncond = _state_slice(merged_next_state, batch_size, 2 * batch_size)

            buttons = pd["buttons"].detach().cpu().numpy()
            camera = pd["camera"].detach().cpu().numpy()
            intra_batch_identical_buttons &= bool(np.array_equal(buttons, np.repeat(buttons[:1], batch_size, axis=0)))
            intra_batch_identical_camera &= bool(np.array_equal(camera, np.repeat(camera[:1], batch_size, axis=0)))
            buttons_out.append(np.squeeze(buttons[0]))
            camera_out.append(np.squeeze(camera[0]))
    elapsed = time.perf_counter() - start

    buttons_out = np.stack(buttons_out)
    camera_out = np.stack(camera_out)

    summary = {
        "batch_size": batch_size,
        "frames": len(frames),
        "elapsed_seconds": elapsed,
        "effective_steps_per_second": float(len(frames) / elapsed),
        "effective_rollouts_per_second_if_all_batch_slots_are_unique": float(batch_size * len(frames) / elapsed),
        "intra_batch_identical_buttons": intra_batch_identical_buttons,
        "intra_batch_identical_camera": intra_batch_identical_camera,
        "buttons_vs_reference": _summarize_against_reference(reference_buttons, buttons_out),
        "camera_vs_reference": _summarize_against_reference(reference_camera, camera_out),
    }

    np.save(os.path.join(args.output_dir, "buttons_first_element.npy"), buttons_out, allow_pickle=True)
    np.save(os.path.join(args.output_dir, "camera_first_element.npy"), camera_out, allow_pickle=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

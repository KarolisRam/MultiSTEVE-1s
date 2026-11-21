#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import cv2

from multisteve.mc_creative import place_tree_commands
from steve1.VPT.agent import ENV_KWARGS
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Capture an in-episode top-down frame after creative placement commands.')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--seed', type=int, default=49)
    p.add_argument('--topdown-height', type=int, default=26, help='Relative Y offset for top-down teleport')
    p.add_argument('--topdown-dx', type=int, default=0, help='Relative X offset for top-down teleport target')
    p.add_argument('--topdown-dz', type=int, default=7, help='Relative Z offset for top-down teleport target')
    p.add_argument('--chat-wait-steps', type=int, default=10)
    p.add_argument('--clear-overhead-leaves', action='store_true',
                   help='Clear nearby leaves around spawn before placing scripted tree targets')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = HumanSurvival(**ENV_KWARGS).make()

    try:
        env.seed(args.seed)
        obs = env.reset()

        placements = [
            'side=left',
            'side=right,tree_type=birch,villager=false',
        ]

        commands = []
        commands += [
            '/gamerule sendCommandFeedback false',
            '/gamerule commandBlockOutput false',
            '/time set day',
            '/weather clear',
        ]
        if args.clear_overhead_leaves:
            commands += [
                '/fill ~-24 ~0 ~-24 ~24 ~40 ~24 minecraft:air replace minecraft:oak_leaves',
                '/fill ~-24 ~0 ~-24 ~24 ~40 ~24 minecraft:air replace minecraft:birch_leaves',
                '/fill ~-24 ~0 ~-24 ~24 ~40 ~24 minecraft:air replace minecraft:spruce_leaves',
            ]
        commands += place_tree_commands(4, 0, 7, tree_type='oak', relative=True, villager=True)
        commands += place_tree_commands(-4, 0, 7, tree_type='birch', relative=True, villager=False)
        dx = f'~{args.topdown_dx}'
        dy = f'~{args.topdown_height}'
        dz = f'~{args.topdown_dz}'
        commands += [
            '/effect give @s minecraft:slow_falling 60 1 true',
            f'/tp @s {dx} {dy} {dz} 0 90',
        ]

        if 'chat' not in env.action_space.noop():
            raise RuntimeError('MineRL action space does not expose chat command channel.')

        for cmd in commands:
            action = env.action_space.noop()
            action['chat'] = cmd
            obs, _, _, _ = env.step(action)

        for _ in range(args.chat_wait_steps):
            action = env.action_space.noop()
            action['chat'] = ''
            obs, _, _, _ = env.step(action)

        pov = obs['pov']
        # Save raw in-episode top-down capture.
        cv2.imwrite(str(out_dir / 'topdown_live.png'), cv2.cvtColor(pov, cv2.COLOR_RGB2BGR))

        # Save central crop for quick visual comparison.
        h, w, _ = pov.shape
        crop = pov[h // 8: h - h // 8, w // 8: w - w // 8]
        cv2.imwrite(str(out_dir / 'topdown_live_crop.png'), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        loc = obs['location_stats']
        keys = sorted(loc.keys())
        with (out_dir / 'location_stats.log').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            writer.writerow([loc[k].item() if hasattr(loc[k], 'item') else loc[k] for k in keys])

        # Keep a command log format compatible with existing overlay script (it parses --placement flags).
        cmdline = (
            f"capture_live_topdown.py --seed {args.seed} "
            f"--placement {placements[0]} "
            f"--placement {placements[1]}"
        )
        (out_dir / 'command.log').write_text(cmdline + '\n', encoding='utf-8')

        # Also save the exact chat commands executed in episode.
        (out_dir / 'chat_commands.log').write_text('\n'.join(commands) + '\n', encoding='utf-8')

    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()

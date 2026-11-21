import argparse
import shutil
import time
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import torch.multiprocessing
import pickle
import gym
import torch
import numpy as np
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed
import os
import wandb
import torch as th
import datetime
import math

from steve1.data.minecraft_dataset import MinecraftDataset, load_sampling
from steve1.MineRLConditionalAgent import MineRLConditionalAgent, configure_optimizers
from steve1.helpers import object_to_torch_and_device, Timer
from steve1.VPT.lib.tree_util import tree_map

from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

# Ensure Cuda determinism. Not necessary in most cases, but can prevent minor randomness in some edge cases.
# Like using a 4090 GPU on CUDA 1.17, which doesn't properly support it... Duh.
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True


# Get the current learning rate for the optimizer (cosine with warmup)
def get_lr(args, num_frames_processed):
    min_lr = 0

    # 1) linear warmup for warmup_iters steps
    if num_frames_processed < args.warmup_frames:
        return args.learning_rate * num_frames_processed / args.warmup_frames
    # 2) if it > lr_decay_iters, return min learning rate
    if num_frames_processed > args.n_frames * 1.1:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (num_frames_processed - args.warmup_frames) / (args.n_frames * 1.1 - args.warmup_frames)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (args.learning_rate - min_lr)


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def get_chunk(x, t, trunc_t):
    if isinstance(x, torch.Tensor):
        return x[:, t:t + trunc_t]
    else:
        return [y[t:t + trunc_t] for y in x]


def force_cudnn_initialization(device):
    s = 32
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=device), torch.zeros(s, s, s, s, device=device))


def resume_training(checkpoint_dir, accelerator):
    accelerator.load_state(checkpoint_dir)
    # open the n_steps file and read the number of steps
    with open(os.path.join(checkpoint_dir, "n_batches.txt"), "r") as f:
        n_batches = int(f.read())
    with open(os.path.join(checkpoint_dir, "best_val_loss.txt"), "r") as f:
        best_val_loss = float(f.read())
    return n_batches, best_val_loss


def emit_checkpoint_schedule(frames_per_batch, exponential_frames, post_exponential_cadence, n_frames):
    def ceil_to_batch(x):
        return ((x + frames_per_batch - 1) // frames_per_batch) * frames_per_batch

    schedule = [0]

    frame = frames_per_batch
    while frame < exponential_frames:
        schedule.append(frame)
        frame *= 2

    k = 0
    while True:
        frame = exponential_frames + k * post_exponential_cadence
        frame = ceil_to_batch(frame)
        if frame > n_frames:
            break
        if frame not in schedule:
            schedule.append(frame)
        k += 1

    final_frame = ceil_to_batch(n_frames)
    if schedule[-1] != final_frame:
        schedule.append(final_frame)

    return schedule


def save_resumable_checkpoint(args, accelerator, n_steps, best_val_loss):
    frames = n_steps * args.batch_size * args.trunc_t * accelerator.num_processes
    print(f"Saving resumable checkpoint to {args.checkpoint_dir} at frame {frames :,}...")
    accelerator.save_state(args.checkpoint_dir)
    with open(os.path.join(args.checkpoint_dir, "n_batches.txt"), "w") as f:
        f.write(str(n_steps * args.trunc_t // args.T))
    with open(os.path.join(args.checkpoint_dir, "best_val_loss.txt"), "w") as f:
        f.write(str(best_val_loss))


def save_shareable_checkpoint(args, accelerator, policy, n_steps):
    frames = n_steps * args.batch_size * args.trunc_t * accelerator.num_processes
    snapshot_path = args.out_weights.replace('.weights', f'_snapshot_{str(frames).zfill(len(str(args.n_frames)))}.weights')
    print(f"Saving shareable checkpoint to {snapshot_path} at frame {frames:,}...")
    state_dict = policy.state_dict()
    state_dict = {k.replace("policy.", ""): v for k, v in state_dict.items()}
    accelerator.save(state_dict, snapshot_path)


def get_wandb_id(checkpoint_dir):
    if not os.path.exists(os.path.join(checkpoint_dir, "wandb_id.txt")):
        wandb_id = wandb.util.generate_id()
        with open(os.path.join(checkpoint_dir, "wandb_id.txt"), "w") as f:
            f.write(wandb_id)
        return wandb_id
    with open(os.path.join(checkpoint_dir, "wandb_id.txt"), "r") as f:
        wandb_id = f.read()
    return wandb_id


def compute_gradient_l2_norm(model):
    """Compute the L2 norm of the gradients of the model parameters.
    
    Precondition: the model parameters must have gradients.
    """
    total_norm = 0
    for i, p in enumerate(model.parameters()):
        if p.grad is not None:
            # print(f'Gradient found for parameter {i}')
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        # else:
        #     print(f'\033[93m' + f'No gradient for parameter {i}: {p}' + '\033[0m')
    total_norm = total_norm ** 0.5
    return total_norm


def compute_weights_l2_norm(model):
    """Compute the L2 norm of the model parameters.
    
    Precondition: the model parameters must have gradients.
    """
    total_norm = 0
    for i, p in enumerate(model.parameters()):
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


class DDPPolicy(torch.nn.Module):

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs, agent_state, firsts, actions):
        """Wraps the policy method to be compatible with DDP"""
        bsz = firsts.shape[0]
        t = firsts.shape[1]
        if agent_state is None:
            agent_state = self.policy.initial_state(bsz)

        pi_distribution, _, new_agent_state = self.policy.get_output_for_observation(
            obs, agent_state, firsts
        )

        pi_distribution = tree_map(lambda x: x.view(bsz, t, -1), pi_distribution)

        log_prob = self.policy.get_logprob_of_action(pi_distribution, actions)

        # Calculate loss
        pi_loss = -log_prob.mean()

        return pi_loss, new_agent_state


def run_validation(policy, val_dataloader, args, device, accelerator):
    agent_state = None
    sum_loss = 0
    n_loss = 0

    val_batch = 0
    total_batch = len(val_dataloader)

    policy.eval()
    with th.no_grad():
        for obs, actions, firsts, _ in val_dataloader:
            val_batch += 1
            T = firsts.shape[1]
            bsz = firsts.shape[0]
            # Chunk into TRUNC_T length sequences
            for t in range(0, T, args.trunc_t):
                # Get the next chunk of frames and actions
                obs_chunk = tree_map(lambda x: get_chunk(x, t, args.trunc_t), obs)
                actions_chunk = tree_map(lambda x: get_chunk(x, t, args.trunc_t), actions)
                firsts_chunk = firsts[:, t:t + args.trunc_t]

                # Convert to torch tensors
                obs_chunk = object_to_torch_and_device(obs_chunk, device)
                actions_chunk = object_to_torch_and_device(actions_chunk, device)
                actions_chunk = tree_map(lambda x: x[:, :, 0], actions_chunk)
                firsts_chunk = object_to_torch_and_device(firsts_chunk, device).view(bsz, args.trunc_t)

                loss, agent_state = policy(obs_chunk, agent_state, firsts_chunk, actions_chunk)

                loss = accelerator.gather(loss).mean()
                sum_loss += loss.item()
                n_loss += 1

            if accelerator.is_main_process:
                print(f"\tFinished validation batch {val_batch}/{total_batch} "
                      f"(bsz={bsz}, val_every_nth={args.val_every_nth})")

    avg_loss = sum_loss / n_loss
    return avg_loss


def main(args):
    set_seed(args.seed)
    out_weights_dir = os.path.dirname(args.out_weights)
    os.makedirs(out_weights_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", 
                              kwargs_handlers=[ddp_kwargs],
                              gradient_accumulation_steps=args.gradient_accumulation_steps)

    if args.restart_from_checkpoint:
        if not os.path.exists(os.path.join(args.checkpoint_dir, "n_batches.txt")):
            if accelerator.is_main_process:
                # Copy the files from restart_from_checkpoint to checkpoint_dir recursively
                if os.path.exists(args.restart_from_checkpoint):
                    print('Copying checkpoint from restart_from_checkpoint...')
                    if os.path.exists(args.checkpoint_dir):
                        shutil.rmtree(args.checkpoint_dir)
                    shutil.copytree(args.restart_from_checkpoint, args.checkpoint_dir)
                    # Wait until the files show up
                    while not os.path.exists(os.path.join(args.checkpoint_dir, "wandb_id.txt")):
                        print('Waiting for files to show up...')
                        time.sleep(5)

    if accelerator.is_main_process:
        wandb_id = get_wandb_id(args.checkpoint_dir)

        accelerator.init_trackers("vpt_finetune",
                                  init_kwargs={
                                      "wandb": {
                                          "resume": "allow",
                                          "id": wandb_id,
                                          "allow_val_change": True,
                                          "name": args.wandb_run_name
                                      }
                                  })
    if accelerator.is_main_process:
        # Add the config to wandb
        wandb.config.update(args, allow_val_change=True)

    # Print the accelerator config
    if accelerator.is_main_process:
        print(accelerator.state)
        print(accelerator.device)
    device = accelerator.device
    force_cudnn_initialization(device)

    # Load datasets
    if accelerator.is_main_process:
        print('Loading dataset...')
    
    train_episodes, val_episodes = load_sampling(args.sampling_dir, args.sampling)
    dataset = MinecraftDataset(train_episodes,
                               args.T,
                               args.min_btwn_goals,
                               args.max_btwn_goals,
                               args.p_uncond,
                               args.data_limit,
                               rng_seed=args.dataset_seed+1)
    if accelerator.is_main_process:
        print(f'Total frames in dataset: {dataset.get_total_frames():,}')
    val_dataset = MinecraftDataset(val_episodes, 
                                   args.T, 
                                   args.min_btwn_goals, 
                                   args.max_btwn_goals, 
                                   args.p_uncond, 
                                   args.data_limit,
                                   every_nth=args.val_every_nth,
                                   rng_seed=args.dataset_seed+2)
    
    if accelerator.is_main_process:
        print(f'Total frames in val dataset: {val_dataset.get_total_frames():,}')

    # Load model
    if accelerator.is_main_process:
        print('Loading model...')
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(args.in_model)
    agent_policy_kwargs["weight_init_seed"] = args.weight_init_seed
    agent_policy_kwargs["bitflip"] = args.bitflip
    agent_policy_kwargs["bitflip_index"] = args.bitflip_index
    agent_policy_kwargs["bitflip_seed"] = args.bitflip_seed
    env = gym.make("MineRLBasaltFindCave-v0")
    agent = MineRLConditionalAgent(env, device=device, policy_kwargs=agent_policy_kwargs,
                                   pi_head_kwargs=agent_pi_head_kwargs)
    if args.in_weights is not None:
        agent.load_weights(args.in_weights)
    env.close()

    policy = DDPPolicy(agent.policy)

    total_trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        # Format in millions (M) of parameters
        print(f"Total trainable parameters: {total_trainable_params / 1e6:.2f}M")

    # Setup optimizer
    optimizer = configure_optimizers(policy, args.weight_decay, args.learning_rate)

    # Setup dataloaders
    generator = torch.Generator()
    generator.manual_seed(args.dataset_seed)
    sampler = RandomSampler(dataset, generator=generator)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                            num_workers=args.num_workers, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, drop_last=True)

    if accelerator.is_main_process:
        print('Preparing accelerator...')
    policy, optimizer, dataloader, val_dataloader = accelerator.prepare(policy, optimizer, dataloader, val_dataloader)

    # Set initial variables
    agent_state = None
    best_val_loss = np.inf

    # Set trn variables
    n_steps, n_batches, sum_loss, n_loss = 0, 0, 0, 0

    timer = Timer('timings')

    # Check if the checkpoint exists (args.checkpoint is empty folder)
    if args.checkpoint_dir:
        # check if n_batches.txt exists in that folder
        if os.path.exists(os.path.join(args.checkpoint_dir, "n_batches.txt")):
            if accelerator.is_main_process:
                print(f'Loading checkpoint from {args.checkpoint_dir}...')
            with accelerator.main_process_first():
                remaining_batches_to_skip, best_val_loss = resume_training(args.checkpoint_dir, accelerator)

            # Set all optimizer states to accelerator.device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(accelerator.device)
        else:
            if accelerator.is_main_process:
                print(f'Checkpoint folder {args.checkpoint_dir} exists '
                      f'but no checkpoint found, starting from scratch...')
                # Save everything
                save_resumable_checkpoint(args, accelerator, n_steps, best_val_loss)
            remaining_batches_to_skip = 0

    episode_chunk_storage = []
    episode_chunks_path = args.out_weights.replace('.weights', f'_data_order.txt')
    shareable_checkpoint_schedule = [0]  # always save initial checkpoint.
    resumable_checkpoint_schedule = []
    if args.shareable_checkpoint_every_n_frames is not None:
        shareable_checkpoint_schedule = emit_checkpoint_schedule(args.batch_size * args.T, args.warmup_frames, args.shareable_checkpoint_every_n_frames, args.n_frames)
        print(f'Shareable checkpoint schedule: {shareable_checkpoint_schedule}')
    if args.resumable_checkpoint_every_n_frames is not None:
        resumable_checkpoint_schedule = emit_checkpoint_schedule(args.batch_size * args.T, 0, args.resumable_checkpoint_every_n_frames, args.n_frames)
        print(f'Resumable checkpoint schedule: {resumable_checkpoint_schedule}')

    real_bsz = args.batch_size * accelerator.num_processes  # accounting for parallelism
    steps_per_batch = args.T // args.trunc_t
    epoch_len = len(dataloader) * steps_per_batch

    frames_per_step = args.trunc_t * real_bsz

    if accelerator.is_main_process:
        # This batch size is num frames per gradient step
        print(f'Batch size in frames: {frames_per_step * args.gradient_accumulation_steps:,}')

    val_freq = args.val_freq_begin

    if accelerator.is_main_process:
        print(f'Starting training at {datetime.datetime.now()}...')

    if remaining_batches_to_skip > 0 and accelerator.is_main_process:
        skip_to_step = remaining_batches_to_skip * steps_per_batch
        print(f'Skipping {remaining_batches_to_skip:,} batches to get to step {skip_to_step:,}...')

    end_training = False
    st = time.time()
    while n_steps * frames_per_step < args.n_frames and not end_training:
        policy.train()
        for obs, actions, firsts, episode_chunks in timer.time_iter(dataloader, 'dataloader'):
            n_batches += 1
            # Using the below skipping logic instead of accelerator.skip_first_batches(), because accelerator doesn't
            # play well with RNG checkpointing and full determinism of resumed training, esp on later epochs.
            # Accelerator prefetches data, which advances the RNG, which is then off on resume.
            if remaining_batches_to_skip > 0:
                remaining_batches_to_skip -= 1
                n_steps += steps_per_batch
                if n_steps * frames_per_step >= args.n_frames:
                    end_training = True
                    break
                continue
            if n_steps * frames_per_step >= args.n_frames:
                end_training = True
                break

            T = firsts.shape[1]
            assert T == args.T
            bsz = firsts.shape[0]
            timer.throughput('frames', T * real_bsz)
            # Chunk into TRUNC_T length sequences
            for t in range(0, T, args.trunc_t):
                with accelerator.accumulate(policy):
                    num_frames_processed = n_steps * frames_per_step

                    # Save initial snapshot
                    if n_steps == 0 and 0 in shareable_checkpoint_schedule:
                        if accelerator.is_main_process:
                            save_shareable_checkpoint(args, accelerator, policy, 0)

                    lr = get_lr(args, num_frames_processed)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    if n_steps >= args.val_freq_switch_steps:
                        val_freq = args.val_freq

                    metrics_log = {}

                    with timer.time('data prep'):
                        # Get the next chunk of frames and actions
                        obs_chunk = tree_map(lambda x: get_chunk(x, t, args.trunc_t), obs)
                        actions_chunk = tree_map(lambda x: get_chunk(x, t, args.trunc_t), actions)
                        firsts_chunk = firsts[:, t:t + args.trunc_t]

                        # Convert to torch tensors
                        obs_chunk = object_to_torch_and_device(obs_chunk, device)
                        actions_chunk = object_to_torch_and_device(actions_chunk, device)
                        actions_chunk = tree_map(lambda x: x[:, :, 0], actions_chunk)
                        firsts_chunk = object_to_torch_and_device(firsts_chunk, device).view(bsz, args.trunc_t)

                    with timer.time('train forward'):
                        loss, new_agent_state = policy(obs_chunk, agent_state, firsts_chunk, actions_chunk)

                    # Backprop
                    with timer.time('train backward'):
                        accelerator.backward(loss)

                    unclipped_gradient_l2_norm = compute_gradient_l2_norm(policy)
                    with timer.time('train clip grad'):
                        accelerator.clip_grad_norm_(policy.parameters(), args.max_grad_norm)

                    with timer.time('train optimizer step'):
                        optimizer.step()

                    # Update agent state
                    agent_state = tree_map(lambda x: x.detach(), new_agent_state)

                    n_steps += 1
                    sum_loss += loss.item()
                    n_loss += 1

                    # Log training metrics when log_freq is reached or when validation is performed
                    if n_steps % args.log_freq == 0 or n_steps % val_freq == 0:
                        avg_loss = sum_loss / n_loss
                        cur_learning_rate = optimizer.param_groups[0]['lr']
                        # Extend accelerator_log with training metrics
                        metrics_log.update({
                            "loss": avg_loss,
                            "step": n_steps,
                            "processed_frames": n_steps * frames_per_step,
                            "epoch": n_steps / epoch_len,
                            "learning_rate": cur_learning_rate,
                            "unclipped_gradient_l2_norm": unclipped_gradient_l2_norm,
                            "grad_l2_norm": compute_gradient_l2_norm(policy),
                            "weights_l2_norm": compute_weights_l2_norm(policy),
                        })
                        sum_loss = 0
                        n_loss = 0

                    with timer.time('train optimizer step'):
                        optimizer.zero_grad()

                    # Run validation code every val_freq trn steps
                    if n_steps % val_freq == 0:
                        if accelerator.is_main_process:
                            print(f'Running validation at step {n_steps}...')
                        with timer.time('validation'):
                            val_loss = run_validation(policy, val_dataloader, args, device, accelerator)
                        metrics_log.update({
                            "val_loss": val_loss,
                        })


                    if metrics_log:
                        metrics_log.update(timer.dict())
                        timer.reset()
                        accelerator.log(metrics_log)
                        if accelerator.is_main_process:
                            print(f'Metrics for step {n_steps}:')
                            print(f'\tCurr DateTime: {datetime.datetime.now()}')
                            for k, v in metrics_log.items():
                                print(f'\t{k}: {v}')

            episode_chunk_storage.append(episode_chunks)

            # Save checkpoints
            frames = n_steps * frames_per_step
            if frames in shareable_checkpoint_schedule:
                if accelerator.is_main_process:
                    save_shareable_checkpoint(args, accelerator, policy, n_steps)
            if frames in resumable_checkpoint_schedule:
                if accelerator.is_main_process:
                    save_resumable_checkpoint(args, accelerator, n_steps, best_val_loss)
                    with open(episode_chunks_path, 'a') as f:
                        for chunks in episode_chunk_storage:
                            for ec in zip(*chunks):
                                f.write(f'{ec[0]},{int(ec[1])},{int(ec[2])}\n')
                    episode_chunk_storage = []

    save_shareable_checkpoint(args, accelerator, policy, n_steps)
    save_resumable_checkpoint(args, accelerator, n_steps, best_val_loss)
    with open(episode_chunks_path, 'a') as f:
        for chunks in episode_chunk_storage:
            for ec in zip(*chunks):
                f.write(f'{ec[0]},{int(ec[1])},{int(ec[2])}\n')
    time_taken = time.time() - st
    total_frames = n_steps * frames_per_step
    print(f'Time taken: {time_taken}. Frames processed: {total_frames}. FPS: {total_frames / time_taken}')

    if accelerator.is_main_process:
        print(f'Running validation at step {n_steps}...')
    with timer.time('validation'):
        val_loss = run_validation(policy, val_dataloader, args, device, accelerator)
    metrics_log.update({
        "val_loss": val_loss,
    })

    accelerator.end_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)  # global random seed, does not impact anything, since all randomness is fully controlled using the below seeds.
    parser.add_argument('--dataset_seed', type=int, default=42)
    parser.add_argument('--weight_init_seed', type=int, default=42)
    parser.add_argument('--bitflip_seed', type=int, default=42)
    parser.add_argument('--bitflip', action='store_true')
    parser.add_argument('--bitflip_index', type=int, default=0)
    parser.add_argument('--sampling', type=str, default='seed0')
    parser.add_argument('--sampling_dir', type=str, default='data/samplings/')
    parser.add_argument('--data_limit', type=int, default=None)
    parser.add_argument('--val_every_nth', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--in_model', type=str, default='data/weights/vpt/2x.model')
    parser.add_argument('--in_weights', type=str, default=None)

    parser.add_argument('--out_weights', type=str,
                        default='data/weights/steve1/trained_with_script.weights')

    parser.add_argument('--T', type=int, default=300)
    parser.add_argument('--p_uncond', type=float, default=0.1)
    parser.add_argument('--min_btwn_goals', type=int, default=15)
    parser.add_argument('--max_btwn_goals', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--trunc_t', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=4e-5)
    parser.add_argument('--n_frames', type=int, default=100_000_000)
    parser.add_argument('--warmup_frames', type=int, default=10_000_000)
    parser.add_argument('--weight_decay', type=float, default=0.039428)
    parser.add_argument('--max_grad_norm', type=float, default=5.0)

    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--shareable_checkpoint_every_n_frames', type=int, default=None)
    parser.add_argument('--resumable_checkpoint_every_n_frames', type=int, default=None)
    parser.add_argument('--val_freq', type=int, default=3500)
    parser.add_argument('--val_freq_begin', type=int, default=3500)
    parser.add_argument('--val_freq_switch_steps', type=int, default=3500)
    parser.add_argument('--save_each_val', type=bool, default=False)

    parser.add_argument('--checkpoint_dir', type=str,
                        default='data/training_checkpoint')
    parser.add_argument('--restart_from_checkpoint', type=str, default=None)

    parser.add_argument('--reset_checkpoint_dir', type=bool, default=False)
    parser.add_argument('--wandb_run_name', type=str, default=None)

    args = parser.parse_args()

    if args.reset_checkpoint_dir:
        if os.path.exists(args.checkpoint_dir):
            shutil.rmtree(args.checkpoint_dir)
        os.makedirs(args.checkpoint_dir)

    main(args)

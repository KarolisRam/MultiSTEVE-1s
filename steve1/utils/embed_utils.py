import torch

from steve1.data.EpisodeStorage import EpisodeStorage


def get_prior_embed(text, mineclip, prior, device, seed=None):
    """Get the embed processed by the prior."""
    use_cuda_amp = str(device).startswith("cuda")
    with torch.cuda.amp.autocast(enabled=use_cuda_amp):
        text_embed = mineclip.encode_text(text).detach().cpu().numpy()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda_amp):
        text_prompt_embed = prior(text_embeddings=torch.tensor(text_embed).float().to(device), seed=seed).cpu().detach().numpy()
    return text_prompt_embed


def get_visual_embed_from_episode(episode_dirpath, timestep):
    """Get the visual embed at the given timestep from the given episode in the dataset. Episode must have been
    saved with EpisodeStorage format (this is how the dataset generation code saves episodes).
    """
    episode = EpisodeStorage(episode_dirpath)
    visual_embeds = episode.load_embeds_attn()
    visual_embed = visual_embeds[timestep]
    return visual_embed

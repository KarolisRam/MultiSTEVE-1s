from setuptools import find_packages, setup


setup(
    name="multisteve1-release",
    version="0.1.0",
    description="MultiSTEVE-1s: A Model Zoo and Interpretability Suite for Instruction-Following Vision Agents",
    author="Karolis Jucys, George Adamopoulos, Özgür Şimşek",
    url="https://sites.google.com/view/multisteve-1s",
    project_urls={
        "Paper": "https://openreview.net/pdf?id=sR3tHXbQxm",
        "Source": "https://github.com/KarolisRam/MultiSTEVE-1s",
        "Model zoo": "https://huggingface.co/MultiSTEVE-1s/MultiSTEVE-1s",
        "Training data": "https://huggingface.co/datasets/MultiSTEVE-1s/steve1-training-data",
        "Seed images": "https://huggingface.co/datasets/MultiSTEVE-1s/seeds",
        "Attention dashboard": "https://multisteve-1s.netlify.app/",
    },
    packages=find_packages(include=["steve1", "steve1.*", "multisteve", "multisteve.*", "tools", "tools.*"]),
)

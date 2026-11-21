import os
from tqdm import tqdm

paths = ['data/dataset_mixed_agents/', 'data/dataset_contractor/']
totaltotal = 0
for path in paths:
    dirs = [f for f in os.listdir(path) if not f.endswith('.py')]
    total = 0

    for d in tqdm(dirs):
        with open(os.path.join(path, d, 'len.txt')) as f:
            n = int(f.readline().strip())
            total += n
    print(path, total, len(dirs))
    totaltotal += total
print(totaltotal)

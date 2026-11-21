import os
import runpy
import sys

import torch


def main():
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.argv = ["steve1/run_agent/run_agent_with_logging_batch_fake_env.py", *sys.argv[1:]]
    runpy.run_path("steve1/run_agent/run_agent_with_logging_batch_fake_env.py", run_name="__main__")


if __name__ == "__main__":
    main()

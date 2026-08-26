"""Assemble the HF Spaces repo and push it.

Usage:
    python scripts/deploy_hf_space.py <your-hf-username>/<space-name>

Requires: `pip install huggingface_hub` and HF_TOKEN env var (or
`huggingface-cli login` beforehand).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGING = ROOT / "spaces" / "_build"


def assemble() -> None:
    """Copy app code + static into a clean build dir next to the Space files."""
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    # Space metadata + Dockerfile + requirements
    shutil.copy(STAGING.parent / "README.md", STAGING / "README.md")
    shutil.copy(STAGING.parent / "Dockerfile", STAGING / "Dockerfile")
    shutil.copy(STAGING.parent / "requirements.txt", STAGING / "requirements.txt")

    # Application code
    shutil.copytree(ROOT / "app", STAGING / "app",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "static", STAGING / "static")
    print(f"Assembled Space repo at {STAGING}")


def push(repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    url = api.create_repo(repo_id=repo_id, repo_type="space",
                          space_sdk="docker", exist_ok=True)
    print(f"Space repo ready: {url}")

    api.upload_folder(
        folder_path=str(STAGING),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy PDF RAG Chatbot",
    )
    print(f"\nDeployed! Track build at: https://huggingface.co/spaces/{repo_id}")
    print("Next: open Space Settings -> Variables and secrets -> add HF_TOKEN")


if __name__ == "__main__":
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        print(__doc__)
        sys.exit(1)
    assemble()
    push(sys.argv[1])

#!/usr/bin/env python3
import argparse
import json

from ftlib.heterocam_config import load_config
from ftlib.heterocam_train import train


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VGGT-Omega camera geometry")
    parser.add_argument("--config", required=True, help="JSON configuration")
    parser.add_argument("--resume", help="latest_training_state.pt from the same experiment")
    args = parser.parse_args()
    result = train(load_config(args.config), args.resume)
    print(json.dumps(result.get("best_passed", result.get("best_candidate", {})), indent=2))


if __name__ == "__main__":
    main()

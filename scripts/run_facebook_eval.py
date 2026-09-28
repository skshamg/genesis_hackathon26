from _bootstrap import *  # noqa: F401,F403

import argparse

from sybil_shield.experiments.real_facebook_eval import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Facebook zero-shot transfer eval.")
    parser.add_argument("--model-id", default=None,
                         help="Registry model id (or 'latest') to evaluate. "
                              "Omit to train fresh at full intensity (120 epochs).")
    parser.add_argument("--model-type", default="gradient_adversarial",
                         help="Registry sub-folder model-id is looked up in.")
    args = parser.parse_args()
    main(model_id=args.model_id, model_type=args.model_type)

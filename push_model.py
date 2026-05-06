#!/usr/bin/env python3

import argparse
import os
import sys

from huggingface_hub import HfApi, create_repo


def parse_args():
	parser = argparse.ArgumentParser(description="Push a local model directory to Hugging Face Hub.")
	parser.add_argument(
		"--input_model_path",
		type=str,
		default="output_models/router_weights_codetask_qwen25_coder_15b",
		help="Path to the local model directory.",
	)
	parser.add_argument(
		"--repo_id",
		type=str,
		required=True,
		help="Target repo on Hugging Face Hub (e.g., username/model-name).",
	)
	parser.add_argument(
		"--token",
		type=str,
		default=None,
		help="Hugging Face token. Defaults to HF_TOKEN env var if omitted.",
	)
	parser.add_argument(
		"--private",
		action="store_true",
		help="Create the repo as private if it does not exist.",
	)
	parser.add_argument(
		"--commit_message",
		type=str,
		default="Upload model",
		help="Commit message for the upload.",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	model_path = args.input_model_path
	if not os.path.isdir(model_path):
		raise FileNotFoundError(f"Input model path does not exist or is not a directory: {model_path}")

	token = args.token or os.environ.get("HF_TOKEN")
	if not token:
		print("Missing Hugging Face token. Provide --token or set HF_TOKEN.", file=sys.stderr)
		sys.exit(1)

	api = HfApi(token=token)
	create_repo(args.repo_id, token=token, private=args.private, exist_ok=True)

	api.upload_folder(
		folder_path=model_path,
		repo_id=args.repo_id,
		commit_message=args.commit_message,
	)
	print(f"Uploaded {model_path} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
	main()

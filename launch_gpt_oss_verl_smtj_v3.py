#!/usr/bin/env python3
"""GPT-OSS-20B SFT+LoRA on SageMaker Training Jobs -- VERL engine, SDK v3 ModelTrainer.

Requires SDK v3: python -m pip install "sagemaker==3.0.1"
(resolves sagemaker-train 1.20.0 / sagemaker-core 2.20.0 -- the versions this was
validated against). SDK 2.x uses a different API and needs the sibling 2.x launcher.

VALIDATED 2026-08-19 on this exact configuration:
  loss 4.947 -> 1.017 over 11 steps, 1 epoch / 1411 multi-turn tool-calling
  records, 8xH100 single node, ~10 min billable, generation-mask self-check
  passed on all ranks, 0 fallback warnings.

Fill in the ---FILL--- constants below; everything else is load-bearing and the
comments say why. Get a green run before simplifying anything.

USAGE:
    python launch_gpt_oss_verl_smtj_v3.py            # dry run, no job created
    python launch_gpt_oss_verl_smtj_v3.py --submit   # actually submits (costs p5 time)
"""

import re
import sys
import tempfile

import boto3
from sagemaker.core import shapes
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.training.configs import Compute, InputData, StoppingCondition
from sagemaker.train import ModelTrainer

PROFILE = None  # None = ambient credentials, which is correct inside Studio
REGION = "us-east-1"  # must match the region the IMAGE_URI below lives in
ROLE_ARN = "---FILL---"  # Studio execution role ARN

# Patched image: stock hyperpod-recipes-verl-0-7-0:verl-v1.1.0-smtj with
# AWSBedrockVerl commit e94ba36d overlaid (multi-turn SFT generation-mask fix).
# The stock image predates that fix and dies on any dataset containing
# role: tool messages, with "TemplateError: Message has tool role, but there was
# no previous assistant message with a tool call!".
# Shared cross-account -- ask the team for the URI, it is not in this repo.
IMAGE_URI = "---FILL---"
RECIPE = "verl-sft-gpt-oss-20b-lora.yaml"
OUTPUT_PATH = "s3://---FILL---/gpt-oss-smoke/output/"
TRAIN_S3 = "s3://---FILL---/gpt-oss-smoke/train/"
# NOTE: in the reference run val was a duplicate of train. Fine for a pipeline
# check; must be fixed before any real training run.
VAL_S3 = "s3://---FILL---/gpt-oss-smoke/val/"

def _localize_results_dir(recipe_path: str) -> str:
    """Rewrite run.results_dir in the recipe text and return a temp recipe path.

    WHY NOT A recipe_overrides ENTRY (which is what the 2.x launcher and v3's own
    from_recipe docstring both do): it crashes on v3.

    Every recipe ships `run.results_dir: ${base_results_dir}/${.name}`, and
    base_results_dir only exists in recipes_collection/config.yaml -- so it is
    always unresolvable when a recipe is loaded standalone, which is why the
    override is mandatory in the first place. But v3's
    _drop_unknown_recipe_overrides() validates each override by walking the base
    recipe and reading `base_recipe[key]`. On run.results_dir that read eagerly
    resolves the interpolation and raises:

        InterpolationKeyError: Interpolation key 'base_results_dir' not found
        full_key: run.results_dir

    So on v3 the single override every hyperpod recipe requires is the one key
    whose validation cannot survive. Setting the value in the recipe text instead
    sidesteps the filter entirely -- the base recipe then holds a plain string,
    nothing needs resolving, and no override is needed.

    This keeps the upstream YAML untouched on disk (pinned to its upstream
    commit); the substitution happens in a temp copy at submit time.
    """
    with open(recipe_path) as f:
        text = f.read()
    patched, n = re.subn(
        r"^(\s*)results_dir:.*$", r"\g<1>results_dir: /opt/ml/model", text, count=1, flags=re.M
    )
    if n != 1:
        raise SystemExit(f"Expected exactly one results_dir line in {recipe_path}, found {n}")
    tmp = tempfile.NamedTemporaryFile(
        mode="w", prefix="recipe_localized_", suffix=".yaml", delete=False
    )
    tmp.write(patched)
    tmp.close()
    return tmp.name


recipe_overrides = {
    "training_config": {
        "data": {
            # Each channel mounts at /opt/ml/input/data/<channel>/. These must
            # match the actual filenames uploaded to TRAIN_S3 / VAL_S3 exactly.
            "train_files": "/opt/ml/input/data/train/---FILL---.jsonl",
            "val_files": "/opt/ml/input/data/val/---FILL---.jsonl",
            # Recipe default is 4096; 16384 suited the multi-turn tool-calling
            # trajectories in the reference run. Drives memory use.
            "max_length": 16384,
            # Do NOT set train_max_samples below train_batch_size (128) -- the
            # trainer refuses with "Cannot train: batch_size (128) is larger than
            # the dataset size". Full 1411 records = ~11 steps/epoch.
        },
        "model": {
            # Recipe default openai/gpt-oss-20b-bf16 does not exist on HF Hub (401).
            # openai/gpt-oss-20b is real but ships mxfp4-quantized MoE experts, while
            # this recipe's engine config (dtype bfloat16, FSDP2) wants plain bf16.
            # unsloth/gpt-oss-20b-BF16 is a dequantized mirror -- fine for a pipeline
            # check; revisit provenance before any real training run.
            "path": "unsloth/gpt-oss-20b-BF16",
            "tokenizer_path": "unsloth/gpt-oss-20b-BF16",
        },
        "trainer": {
            "total_epochs": 1,
        },
    },
}


def main():
    submit = "--submit" in sys.argv

    session = Session(boto_session=boto3.Session(profile_name=PROFILE, region_name=REGION))

    trainer = ModelTrainer.from_recipe(
        training_recipe=_localize_results_dir(RECIPE),
        recipe_overrides=recipe_overrides,
        compute=Compute(instance_type="ml.p5.48xlarge", instance_count=1),
        # 2.x called this image_uri. Not auto-resolved for this engine either way.
        training_image=IMAGE_URI,
        output_data_config=shapes.OutputDataConfig(s3_output_path=OUTPUT_PATH),
        # 2.x called this max_run=1800.
        stopping_condition=StoppingCondition(max_runtime_in_seconds=1800),
        sagemaker_session=session,
        role=ROLE_ARN,
        base_job_name="v3-gpt-oss-20b-sft-verl-lora",
    )
    print("ModelTrainer constructed -- recipe loaded, overrides merged, results_dir resolved.")

    inputs = [
        InputData(channel_name="train", data_source=TRAIN_S3),
        InputData(channel_name="val", data_source=VAL_S3),
    ]

    if not submit:
        trainer.train(input_data_config=inputs, dry_run=True)
        print("\nDRY RUN ok -- nothing submitted. Re-run with --submit to launch.")
        return

    trainer.train(input_data_config=inputs, wait=False)
    print("\nSubmitted. Compare against the 2.x baseline:")
    print("  GOOD: 'genmask: installed generation-masked template ...; self-check passed.'")
    print("        loss ~4.96 -> ~1.01 over 11 steps, mask ratio (mr:) steady ~3.0")
    print("  BAD:  'Message has tool role, but there was no previous assistant message'")
    print("        'using per-turn fallback' / loss flat or zero")
    print(f"\n  aws logs tail /aws/sagemaker/TrainingJobs --follow \\")
    print(f"      --region {REGION} --log-stream-name-prefix <job-name>")


if __name__ == "__main__":
    main()

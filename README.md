# GPT-OSS-20B SFT + LoRA smoke test on SageMaker Training Jobs

Submits a single-node, 8×H100 SFT + LoRA job for GPT-OSS-20B on multi-turn
tool-calling data, using the VERL engine via a `sagemaker-hyperpod-recipes` recipe.

This exact configuration was validated on 2026-08-19: **loss 4.947 → 1.017 over 11
steps** (1 epoch, 1411 records), ~10 min billable, generation-mask self-check passed
on all ranks, zero fallback warnings.

## Contents

| File | Notes |
|---|---|
| `launch_gpt_oss_verl_smtj_v3.py` | The launcher. SDK v3 `ModelTrainer`. Only the `---FILL---` constants need changing. |
| `verl-sft-gpt-oss-20b-lora.yaml` | The recipe, vendored from `aws/sagemaker-hyperpod-recipes` at commit `20ea8f4551cd540b5b023b25d41ab414b16fe493`. Unmodified. |
| `validate_chat_template_contract.py` | Pre-flights the training JSONL against the chat-template contract. Stdlib only. |

The training container image URI is **not** in this repo — ask the team for it.

## Setup

The launcher needs SageMaker SDK **v3**. Studio images ship an older
`sagemaker-train`, so install into a clean venv (no `--system-site-packages`, so
nothing mixes with the image's packages):

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install "sagemaker==3.0.1"
python -c "import importlib.metadata as m; print(m.version('sagemaker'), m.version('sagemaker-train'))"
# expect: 3.0.1 1.20.0  -- the versions this was validated against
```

Use `python -m pip`, not `pip` — in Studio those can be different interpreters.

## Fill in

In `launch_gpt_oss_verl_smtj_v3.py`:

| Constant | Value |
|---|---|
| `ROLE_ARN` | Studio execution role ARN. Needs S3 read on the data, write to output, and ECR pull on the shared image repo. |
| `IMAGE_URI` | From the team. |
| `TRAIN_S3` / `VAL_S3` | S3 prefixes, one JSONL file each. |
| `OUTPUT_PATH` | Where artifacts land. |
| `train_files` / `val_files` | The **filenames** inside those prefixes. Channels mount at `/opt/ml/input/data/<channel>/`, so these must match exactly. |

`REGION` must match the region of `IMAGE_URI` — SageMaker requires the image and
the job to be in the same region.

**If the dataset has fewer than 128 records**, add `"train_batch_size": 32` to the
`data` block. The recipe defaults to 128 and the trainer hard-fails with
`Cannot train: batch_size (128) is larger than the dataset size`. Use a multiple of 8.

## Data format

Plain OpenAI-format JSONL, one full conversation per line:
`{"messages": [...], "tools": [...]}`. Two things crash the GPT-OSS chat template
if you get them wrong:

- `tool_calls[].function.arguments` must be a **parsed JSON object**, not a JSON string.
- Delete any `"default": null` keys from tool parameter specs. The template's
  tool-schema renderer does raw string concatenation on an enum parameter's default
  instead of using `|tojson`, so a JSON null raises
  `TypeError: can only concatenate str (not "NoneType") to str`.

Check before submitting:

```bash
python validate_chat_template_contract.py <your>.jsonl --max-len 16384
```

## Run

```bash
python launch_gpt_oss_verl_smtj_v3.py            # dry run -- creates nothing
python launch_gpt_oss_verl_smtj_v3.py --submit   # submits; costs p5 time
```

The dry run does a real IAM permission check, so it catches role problems before
you queue for a GPU. Note v3 logs `"<job-name> job submitted."` even on a dry run —
nothing is created; that's an upstream logging bug.

## Reading the logs

```bash
aws logs tail /aws/sagemaker/TrainingJobs --follow \
    --region us-east-1 --log-stream-name-prefix <job-name>
```

**Good:** `genmask: installed generation-masked template ...; self-check passed.`,
then falling `train/loss` with `train/loss_mask_ratio` holding steady (~3.0 in the
reference run — a steady ratio is what shows assistant tokens are actually being
graded rather than silently masked to nothing).

**Bad:**

| Symptom | Cause |
|---|---|
| `Message has tool role, but there was no previous assistant message with a tool call!` | Stock image, not the patched one. |
| `using per-turn fallback` / `assistant_masks present but empty` | Mask inactive; loss will look flat. |
| `TypeError: can only concatenate str (not "NoneType") to str` | A `"default": null` survived in the tool specs. |
| `ValueError: Invalid role 'tool' at message index N` | That's the LLMFT engine — wrong recipe file. |
| `Cannot train: batch_size (128) is larger than the dataset size` | Set `train_batch_size`. |
| Hang or connection error at model load | No HuggingFace egress from the training subnet; weights need staging to S3. |
| `AccessDenied` pulling the image | Repo policy is open but the execution role's own policy lacks `ecr:BatchGetImage` / `GetDownloadUrlForLayer` / `BatchCheckLayerAvailability` / `GetAuthorizationToken`. |

## Known caveats

- `model.path` points at `unsloth/gpt-oss-20b-BF16`, a third-party dequantized
  mirror. The recipe's own default (`openai/gpt-oss-20b-bf16`) does not exist on HF
  Hub, and `openai/gpt-oss-20b` ships mxfp4-quantized experts incompatible with this
  recipe's `bfloat16` + FSDP2 config. Fine for a smoke test; review provenance before
  production training.
- `run.results_dir` is baked into a temp copy of the recipe rather than passed as a
  `recipe_overrides` entry. See the docstring on `_localize_results_dir` — the
  documented override crashes on SDK v3.
- Validation used val data duplicated from train. Fix before real training.

# Self-Hosted GPU CI

rvLLM now includes two manual GitHub Actions workflows intended for self-hosted Linux/NVIDIA runners:

- `GPU Smoke`
- `GPU Bench`

These workflows are designed for fork-side validation first, especially when you want to verify CUDA-path behavior before proposing upstream performance claims.

## Why self-hosted

rvLLM's real serving path is CUDA-first. The most important checks for this repo are:

- direct benchmark correctness on a real NVIDIA GPU
- CUDA graph replay behavior
- HTTP benchmark behavior against a live server
- artifact capture for reproducible follow-up work

GitHub-hosted CPU runners are still used for normal `cargo check` / `cargo test`, but the GPU workflows expect your own runner labels and hardware.

## Workflow Inputs

Both workflows are `workflow_dispatch` only and accept a configurable `runs_on` input. The default value is:

```json
["self-hosted", "linux", "x64", "nvidia", "cuda"]
```

Match your runner labels to that set, or override the input when launching the workflow.

## GPU Smoke

`GPU Smoke` is the fastest CUDA-path check. It:

1. checks out the repo
2. installs the Rust toolchain
3. builds `rvllm` with CUDA features
4. runs `rvllm info`
5. runs a minimal direct benchmark
6. uploads logs and benchmark JSON as artifacts

Use it when:

- bringing up a new runner
- confirming CUDA build viability on a new machine
- smoke-checking a branch before a longer benchmark

## GPU Bench

`GPU Bench` is the fuller manual benchmark path. It:

1. builds `rvllm` with CUDA features
2. captures runner and GPU diagnostics
3. runs the direct benchmark command
4. starts a local `rvllm serve` process
5. runs the HTTP benchmark client against that server
6. uploads logs, JSON outputs, and GPU diagnostics as artifacts
7. writes a concise job summary to the Actions UI

Use it when:

- validating a hot-path change
- comparing throughput across branches
- collecting artifacts for an issue, RFC, or PR description

## Required Secret

If your model download path requires Hugging Face authentication, add:

- `HF_TOKEN`

to the repository or fork secrets used by the workflow.

## Suggested Launch Order

1. run `GPU Smoke`
2. confirm the runner, CUDA toolchain, and model access are healthy
3. run `GPU Bench`
4. download the artifacts and compare the benchmark JSON across branches

## Notes

- The workflows do not provision GPU drivers or CUDA for you. The runner must already be configured.
- These workflows are intentionally manual-only to avoid burning GPU time on every push.
- For fork-first experimentation, point the runner at your fork before exposing it to upstream repositories.

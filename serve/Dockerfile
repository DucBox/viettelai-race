# Submission image for Viettel AI Race — Track 3.
# Bakes model weights into the image at build time (no runtime pull —
# competition anti-cheat rule forbids fetching weights from the network
# while serving).
#
# BTC's chấm instance is a MIG H200 (Ubuntu 22.04 / CUDA 12.x) — a standard
# x86_64 datacenter GPU box, not Grace Hopper (ARM). The `--platform` pin below
# forces linux/amd64 regardless of what CPU you build this on (e.g. Apple
# Silicon Macs default to arm64 and would silently produce an image that
# cannot run on the grading server).
#
# Build from the serve/ directory so the build context only contains
# what's needed (models/, not the whole repo):
#   cd serve
#   docker build --platform linux/amd64 -t <dockerhub-user>/qwen35-2b-race:v1 .
#   docker push <dockerhub-user>/qwen35-2b-race:v1
#
# Then point docker-compose-baseline.yml's `image:` at that tag and
# submit that compose file through the Portal.

FROM --platform=linux/amd64 vllm/vllm-openai:v0.22.1

COPY models/qwen3.5-2b/ /model/

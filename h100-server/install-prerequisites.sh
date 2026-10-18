#!/usr/bin/env bash
set -euo pipefail

# Run as azureuser on the existing Ubuntu 22.04 H100 VM.
[[ $(hostname) == gemma-h100-aue ]]
[[ $(id -un) == azureuser ]]
. /etc/os-release
[[ $ID == ubuntu && $VERSION_ID == 22.04 ]]

sudo apt-get update
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential pkg-config libssl-dev python3-venv curl ca-certificates \
  jq rsync "linux-headers-$(uname -r)"

curl --fail --location --retry 3 \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
  --output "$HOME/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i "$HOME/cuda-keyring_1.1-1_all.deb"
sudo apt-get update
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
  nvidia-open=615.71.09-2ubuntu1 cuda-toolkit-12-8=12.8.2-1
sudo modprobe nvidia
nvidia-smi

curl --fail --location --retry 3 https://sh.rustup.rs --output "$HOME/rustup-init.sh"
bash "$HOME/rustup-init.sh" -y --profile minimal --default-toolchain 1.96.0
"$HOME/.cargo/bin/rustc" --version
/usr/local/cuda-12.8/bin/nvcc --version

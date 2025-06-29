cd /data/mmh/nano-vllm

export NCCL_DEBUG=WARNING


# fix bug: the nccl works with docker in interactivate shell, but has problems in non-interactivate mode, after diff the environment variables, we should do this things
# copied from  /etc/bash.bashrc in container.
source /etc/profile.d/env.sh
source /opt/dtk-24.04.3/env.sh
source /opt/dtk/env.sh
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_LAUNCH_MODE=GROUP
export NCCL_MAX_NCHANNELS=128
export NCCL_MIN_NCHANNELS=128
export NCCL_P2P_LEVEL=SYS
# end copied from /etc/bash.bashrc



export \
VLLM_LOGGING_LEVEL=DEBUG \
VLLM_USE_TRITON_FLASH_ATTN=1 \
NCCL_ALGO=Ring \
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
FLASH_ATTENTION_SKIP_CK_BUILD=TRUE \
FLASH_ATTENTION_PRINT_PARAM=1 \
NCCL_MAX_NCHANNELS=128 \
NCCL_MIN_NCHANNELS=128 \
NCCL_NET_GDR_READ=0 \
NCCL_IB_HCA=mlx5_0:1,mlx5_1:1 \
NCCL_ALGO=Ring \
HSA_FORCE_FINE_GRAIN_PCIE=1 \
NCCL_MIN_P2P_NCHANNELS=128 \
NCCL_NCHANNELS_PER_PEER=128 \
NCCL_SOCKET_IFNAME=ibp1s0 \
NCCL_IB_TIMEOUT=20 \
VLLM_TORCH_PROFILER_DIR=/data/mmh/vllm_tracing

pip install /data/fh/triton-3.1.0-cp310-cp310-linux_x86_64.whl
ln -s /opt/dtk /opt/rocm


pkill pt_main
sleep $1
python example.py --node-id=$1 --node-num=8
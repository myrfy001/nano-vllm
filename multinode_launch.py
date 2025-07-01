from fabric import Connection, Config, ThreadingGroup
from threading import Thread

shutdown_hosts = [(f"10.18.17.{i+141}", f"h02r3n{i:02d}") for i in range(3, 3+15)]
hosts = [(f"10.18.17.{i+141}", f"h02r3n{i:02d}") for i in range(3, 3+2)]
password = "XaNjj@##Apir!"


def add_docker_to_user_group(node_id, ip_addr, user_name):
    config = Config(overrides={'sudo': {'password': password}})
    c = Connection(ip_addr, user=user_name, connect_kwargs={"password": password}, config=config)
    c.sudo("usermod -aG docker $USER")
    c.close()

def stop_all(node_id, ip_addr, user_name):
    config = Config(overrides={'sudo': {'password': password}})
    c = Connection(ip_addr, user=user_name, connect_kwargs={"password": password}, config=config)
    r = c.run(f"""docker stop ds_pp || true && \
              docker rm ds_pp || true && \
              docker ps | grep ds_pp
              """)
    c.close()


def make_sure_docker_stopped(ip_addr, user_name):
    config = Config(overrides={'sudo': {'password': password}})
    c = Connection(ip_addr, user=user_name, connect_kwargs={"password": password}, config=config)
    r = c.run(f"""
              docker ps -a
              """)
    print(f"make_sure_docker_stopped {ip_addr} = {r.stdout}")
    assert "ds_pp" not in r.stdout
    c.close()


def run_model(node_id, ip_addr, user_name):
    config = Config(overrides={'sudo': {'password': password}})
    c = Connection(ip_addr, user=user_name, connect_kwargs={"password": password}, config=config)
    r = c.run(f"""
                export IMAGE=image.sourcefind.cn:5000/dcu/admin/base/custom:vllm0.8.5-ubuntu22.04-dtk25.04-rc7-das1.5-py3.10-20250611-fixpy-rocblas0611-rc1
              
                docker run \
                --name ds_pp \
                --rm \
                --privileged \
                --network=host \
                --ipc=host \
                --shm-size=16G \
                -v /opt/hyhal:/opt/hyhal \
                -v /htxjj:/data \
                --group-add video \
                --cap-add=SYS_ADMIN \
                --security-opt seccomp=unconfined \
                --device=/dev/kfd \
                --device=/dev/mkfd \
                --device=/dev/dri \
                --device=/dev/infiniband/uverbs0 \
                --device=/dev/infiniband/uverbs1 \
                --device=/dev/infiniband/uverbs2 \
                --device=/dev/infiniband/uverbs3 \
                --device=/dev/infiniband/rdma_cm \
                --cap-add=IPC_LOCK \
                --cap-add=SYS_PTRACE \
                -e XINFERENCE_MODEL_SRC=modelscope \
                -e XINFERENCE_HOME=/data/xinference \
                -e VLLM_USE_MODELSCOPE=1 \
                -e VLLM_USE_TRITON_FLASH_ATTN=1 \
                -e VLLM_USE_TRITON_PREFIX_FLASH_ATTN=1 \
                -e VLLM_USE_TRITON_OPT_MLA=1 \
                -e NCCL_ALGO=Ring \
                -e NCCL_MAX_NCHANNELS=16 \
                -e NCCL_MIN_NCHANNELS=16 \
                -e NCCL_NET_GDR_READ=0 \
                -e NCCL_IB_HCA=mlx5_0:1,mlx5_1:1 \
                -e NCCL_MIN_P2P_NCHANNELS=16 \
                -e NCCL_NCHANNELS_PER_PEER=16 \
                -e NCCL_SOCKET_IFNAME=ibp1s0 \
                -e NCCL_IB_TIMEOUT=20 \
                -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
                -e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
                -e FLASH_ATTENTION_SKIP_CK_BUILD=TRUE \
                -e FLASH_ATTENTION_PRINT_PARAM=1 \
                -e TRITON_PRINT_AUTOTUNING=1 \
                $IMAGE \
                /bin/bash /data/mmh/nano-vllm/docker_entry_point.sh {node_id}
              """)
    c.close()


threads = []
for node_id, (ip_addr, user_name) in enumerate(shutdown_hosts):
    node_id = node_id + 1
    threads.append(Thread(target=stop_all, args=(node_id, ip_addr, user_name)))
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()


for node_id, (ip_addr, user_name) in enumerate(shutdown_hosts):
    make_sure_docker_stopped(ip_addr, user_name)
    

threads = []
for node_id, (ip_addr, user_name) in enumerate(hosts):
    node_id = node_id + 1
    threads.append(Thread(target=run_model, args=(node_id, ip_addr, user_name)))
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
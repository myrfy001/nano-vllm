from fabric import Connection, Config, ThreadingGroup

target_ip = "47.108.189.36"
target_port = "60005"
password = "PzsyS2020_vip"


def scp_copy(ip_addr, port, user_name, password, remote_path, local_path):
    config = Config(overrides={'sudo': {'password': password}})
    c = Connection(ip_addr, user=user_name, port=port,connect_kwargs={"password": password}, config=config)
    c.get(remote_path, local_path)
    c.close()

if __name__ == "__main__":
    scp_copy(target_ip, target_port, "root", password, "/data/model/debug/0.safetensors", "./")
#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import threading
import time
import xmlrpc.client
from xmlrpc.server import SimpleXMLRPCServer
from argparse import ArgumentParser
from fabric import Connection, Config, ThreadingGroup

def start_rpc_server(fns, host='localhost', port=8001):
    server = SimpleXMLRPCServer((host, port), allow_none=True)

    for fn_name, fn in fns.items():
        server.register_function(fn, fn_name)

    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    print(f"[Rank 0 RPC] 服务器已在 http://{host}:{port} 上启动。")


def call_prefill(prompt: str = '你是谁', host: str = '47.108.189.36', port: int = 60004) -> str:
    proxy = xmlrpc.client.ServerProxy(f'http://{host}:{port}/')
    try:
        # start_time = time.time()

        # 这个调用会阻塞，直到服务端完成任务并返回结果
        file_path = proxy.start_task(prompt)

        # end_time = time.time()
        # print("\n" + "="*30)
        # print("           调用完成！")
        # print("="*30)
        # print(f"服务端返回结果: '{result}'")
        # print(f"客户端实际阻塞时间: {end_time - start_time:.2f} 秒")
        return file_path.replace('root', 'data')

    except ConnectionRefusedError:
        print("连接失败！请确保服务端正在运行。")
    except Exception as e:
        print(f"发生错误: {e}")


def scp_copy(remote_path, local_path = '/data/debug', ip_addr = '47.108.189.36', port = '60005', user_name = 'root', password = 'PzsyS2020_vip'):
    config = Config(overrides={'sudo': {'password': password}})
    c = Connection(ip_addr, user=user_name, port=port,connect_kwargs={"password": password}, config=config)
    c.get(remote_path, local_path)
    c.close()


def main(prompt: str, host: str, port: int):
    proxy = xmlrpc.client.ServerProxy(f'http://{host}:{port}/')
    try:
        start_time = time.time()

        # 这个调用会阻塞，直到服务端完成任务并返回结果
        result = proxy.start_task(prompt)

        end_time = time.time()

        print("\n" + "="*30)
        print("           调用完成！")
        print("="*30)
        print(f"服务端返回结果: '{result}'")
        print(f"客户端实际阻塞时间: {end_time - start_time:.2f} 秒")

    except ConnectionRefusedError:
        print("连接失败！请确保服务端正在运行。")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--prompt", type=str, default="你是谁")
    args = parser.parse_args()
    main(args.prompt, args.host, args.port)


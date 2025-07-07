from glob import glob
import os
import torch
import time
from safetensors import safe_open
from threading import Thread
import shutil

kvcahce_sync_dir = "/data/debug1/"


class KvCacheMonitor:
    
    def __init__(self, kvcache: torch.Tensor):
        self.empty_directory(kvcahce_sync_dir)
        self.kvcache = kvcache
        self.thread = Thread(target=self.monitor_kvcache_change)
        self.thread.start()

    def monitor_kvcache_change(self):

        cur_kvcache_id = 0
        while True:
            time.sleep(1)
            all_files = list(sorted(glob(os.path.join(kvcahce_sync_dir, "*.safetensors"))))
            if len(all_files) == 0:
                continue

            newest_fn = all_files[-1]
            fn_id = int(newest_fn.split("/")[-1].split(".")[0])

            if fn_id > cur_kvcache_id:
                cur_kvcache_id = fn_id

                time.sleep(5)
                print("found new kvcache, begin load kvcache")
                with safe_open(newest_fn, "pt", "cpu") as safetensor_file:
                    metadata = safetensor_file.metadata()
                    seqlen, start_pos = int(metadata['seqlen']), int(metadata['start_pos'])
                    assert start_pos == 0
                    loaded_kv_cache = safetensor_file.get_tensor("kv_caches")
                    self.kvcache[:, :seqlen] = loaded_kv_cache.to(self.kvcache.device)
                print("finish load kvcache")


    def empty_directory(self, directory):
        for root, dirs, files in os.walk(directory):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))


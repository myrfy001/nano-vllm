import triton
import triton.language as tl
import torch

from typing import Any, Dict, List, Optional, Tuple
from torch.profiler import ProfilerActivity
import random


@triton.jit
def serialize_context(
        is_prefill,
        slot_mapping_ptr, slot_mapping_size: tl.constexpr,
        context_lens_ptr, context_lens_size: tl.constexpr,
        block_tables_ptr, block_tables_size: tl.constexpr,
        hidden_state_ptr, hidden_state_size: tl.constexpr,
        positions_ptr, positions_size: tl.constexpr,
        buf_ptr,
        buf_stride: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
    # each row is 7168 * 2 bytes

    buf_fp16_elements = buf_stride // 2
    
    buf_ptr_as_u16 = buf_ptr.cast(tl.pointer_type(tl.uint16))

    tl.store(buf_ptr_as_u16, is_prefill)
    tl.store(buf_ptr_as_u16 + 1, slot_mapping_size)
    tl.store(buf_ptr_as_u16 + 2, context_lens_size)
    tl.store(buf_ptr_as_u16 + 3, block_tables_size)
    tl.store(buf_ptr_as_u16 + 4, hidden_state_size)
    tl.store(buf_ptr_as_u16 + 5, positions_size)


    buf_ptr += buf_stride
    buf_ptr_as_i32 = buf_ptr.cast(tl.pointer_type(tl.int32))
    for block_idx in range(0, tl.cdiv(slot_mapping_size, BLOCK_SIZE)):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        masks = offsets < slot_mapping_size
        load_addr = slot_mapping_ptr + offsets
        store_addr = buf_ptr_as_i32 + offsets
        loaded_int32 = tl.load(load_addr, mask=masks)
        tl.store(store_addr, loaded_int32, mask=masks)

    buf_ptr += buf_stride
    buf_ptr_as_i32 = buf_ptr.cast(tl.pointer_type(tl.int32))
    for block_idx in range(0, tl.cdiv(context_lens_size, BLOCK_SIZE)):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        masks = offsets < context_lens_size
        load_addr = context_lens_ptr + offsets
        store_addr = buf_ptr_as_i32 + offsets
        loaded_int32 = tl.load(load_addr, mask=masks)
        tl.store(store_addr, loaded_int32, mask=masks)

    buf_ptr += buf_stride
    buf_ptr_as_i32 = buf_ptr.cast(tl.pointer_type(tl.int32))
    for block_idx in range(0, tl.cdiv(block_tables_size, BLOCK_SIZE)):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        masks = offsets < block_tables_size
        load_addr = block_tables_ptr + offsets
        store_addr = buf_ptr_as_i32 + offsets
        loaded_int32 = tl.load(load_addr, mask=masks)
        tl.store(store_addr, loaded_int32, mask=masks)

    buf_ptr += buf_stride
    buf_ptr_as_i64 = buf_ptr.cast(tl.pointer_type(tl.int64))
    for block_idx in range(0, tl.cdiv(positions_size, BLOCK_SIZE)):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        masks = offsets < positions_size
        load_addr = positions_ptr + offsets
        store_addr = buf_ptr_as_i64 + offsets
        loaded_int64 = tl.load(load_addr, mask=masks)
        tl.store(store_addr, loaded_int64, mask=masks)

    buf_ptr += buf_stride
    buf_ptr_as_fp16 = buf_ptr.cast(tl.pointer_type(tl.float16))
    # TODO: use grid to handle seq_idx dim
    for seq_idx in range(0, hidden_state_size):
        for block_idx in range(0, tl.cdiv(buf_fp16_elements, BLOCK_SIZE)):
            offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            masks = offsets < buf_fp16_elements
            load_addr = hidden_state_ptr + seq_idx * buf_fp16_elements + offsets
            store_addr = buf_ptr_as_fp16 + seq_idx * buf_fp16_elements + offsets
            loaded_fp16 = tl.load(load_addr, mask=masks)
            tl.store(store_addr, loaded_fp16, mask=masks)


@triton.jit
def deserialize_context(
        buf_ptr,
        buf_stride: tl.constexpr,
        meta_ptr,
        slot_mapping_ptr, slot_mapping_size: tl.constexpr,
        context_lens_ptr, context_lens_size: tl.constexpr,
        block_tables_ptr, block_tables_size: tl.constexpr,
        hidden_state_ptr, hidden_state_size: tl.constexpr,
        positions_ptr, positions_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
    # each row is 7168 * 2 bytes

    buf_fp16_elements = buf_stride // 2
    
    buf_ptr_as_u16 = buf_ptr.cast(tl.pointer_type(tl.uint16))

    is_prefill_val = tl.load(buf_ptr_as_u16)
    slot_mapping_len = tl.load(buf_ptr_as_u16 + 1)
    context_lens_len = tl.load(buf_ptr_as_u16 + 2)
    block_tables_len = tl.load(buf_ptr_as_u16 + 3)
    hidden_state_len = tl.load(buf_ptr_as_u16 + 4)
    positions_len = tl.load(buf_ptr_as_u16 + 5)

    tl.store(meta_ptr, is_prefill_val)
    tl.store(meta_ptr+1, slot_mapping_len)
    tl.store(meta_ptr+2, context_lens_len)
    tl.store(meta_ptr+3, block_tables_len)
    tl.store(meta_ptr+4, hidden_state_len)
    tl.store(meta_ptr+5, positions_len)


    buf_ptr += buf_stride
    buf_ptr_as_i32 = buf_ptr.cast(tl.pointer_type(tl.int32))
    for block_idx in range(0, tl.cdiv(slot_mapping_size, BLOCK_SIZE)):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        masks = offsets < slot_mapping_len
        loaded_elements = tl.load(buf_ptr_as_i32 + offsets, mask=masks)
        tl.store(slot_mapping_ptr + offsets, loaded_elements, mask=masks)


    buf_ptr += buf_stride
    buf_ptr_as_i32 = buf_ptr.cast(tl.pointer_type(tl.int32))
    for block_idx in range(0, tl.cdiv(context_lens_size, BLOCK_SIZE)):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        masks = offsets < context_lens_len
        loaded_elements = tl.load(buf_ptr_as_i32 + offsets, mask=masks)
        tl.store(context_lens_ptr + offsets, loaded_elements, mask=masks)


    buf_ptr += buf_stride
    buf_ptr_as_i32 = buf_ptr.cast(tl.pointer_type(tl.int32))
    for block_idx in range(0, tl.cdiv(block_tables_size, BLOCK_SIZE)):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        masks = offsets < block_tables_len
        loaded_elements = tl.load(buf_ptr_as_i32 + offsets, mask=masks)
        tl.store(block_tables_ptr + offsets, loaded_elements, mask=masks)

    buf_ptr += buf_stride
    buf_ptr_as_i64 = buf_ptr.cast(tl.pointer_type(tl.int64))
    for block_idx in range(0, tl.cdiv(positions_size, BLOCK_SIZE)):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        masks = offsets < positions_len
        loaded_elements = tl.load(buf_ptr_as_i64 + offsets, mask=masks)
        tl.store(positions_ptr + offsets, loaded_elements, mask=masks)

    buf_ptr += buf_stride
    buf_ptr_as_fp16 = buf_ptr.cast(tl.pointer_type(tl.float16))
    # TODO: use grid to handle seq_idx dim
    for seq_idx in range(0, hidden_state_len):
        for block_idx in range(0, tl.cdiv(buf_fp16_elements, BLOCK_SIZE)):
            offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            masks = offsets < buf_fp16_elements
            loaded_elements = tl.load(buf_ptr_as_fp16 + seq_idx * buf_fp16_elements + offsets, mask=masks)
            tl.store(hidden_state_ptr + seq_idx * buf_fp16_elements + offsets, loaded_elements, mask=masks)


def invoke_serialize_context_kernel(
        is_prefill: bool,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        hidden_state: torch.Tensor,
        positions: torch.Tensor,
        buf: torch.Tensor
    ):

    assert buf.dtype == torch.uint8 and buf.stride(0) == 7168 * 2
    assert slot_mapping.size(0) <= 2048
    assert positions.size(0) <= 1024  # this is int64
    assert hidden_state.is_contiguous()


    serialize_context[(1,)](
        int(is_prefill),
        slot_mapping, slot_mapping.size(0),
        context_lens, context_lens.size(0),
        block_tables, block_tables.size(1),
        hidden_state, hidden_state.size(0),
        positions, positions.size(0),
        buf, buf.stride(0),
        BLOCK_SIZE=1024,
        num_warps=16
    )

def invoke_deserialize_context_kernel(
        buf: torch.Tensor,
        meta: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        hidden_state: torch.Tensor,
        positions: torch.Tensor,
    ):
    assert buf.is_contiguous()
    assert hidden_state.is_contiguous()

    deserialize_context[(1,)](
        buf, buf.stride(0),
        meta,
        slot_mapping, slot_mapping.size(0),
        context_lens, context_lens.size(0),
        block_tables, block_tables.size(1),
        hidden_state, hidden_state.size(0),
        positions, positions.size(0),
        BLOCK_SIZE=1024
    )


def test():
    torch.set_printoptions(edgeitems=16)

    src_slot_mapping_size = random.randint(1, 2048)
    src_context_lens_size = random.randint(1, 2048)
    src_block_tables_size = random.randint(1, 2048)
    src_hidden_state_size = 256
    src_positions_size = random.randint(1, 1024)

    src_slot_mapping: torch.Tensor = torch.randint(1, 100, (src_slot_mapping_size,), dtype=torch.int32).cuda()
    src_context_lens: torch.Tensor = torch.randint(1, 100, (src_context_lens_size,), dtype=torch.int32).cuda()
    src_block_tables: torch.Tensor = torch.randint(1, 100, (src_block_tables_size,), dtype=torch.int32).cuda()
    src_hidden_state: torch.Tensor = torch.rand((src_hidden_state_size, 7168), dtype=torch.float16).cuda()
    src_positions: torch.Tensor = torch.randint(1, 100, (src_positions_size,), dtype=torch.int64).cuda()

    dst_slot_mapping: torch.Tensor = torch.randint(1, 100, (2048,), dtype=torch.int32).cuda()
    dst_context_lens: torch.Tensor = torch.randint(1, 100, (2048,), dtype=torch.int32).cuda()
    dst_block_tables: torch.Tensor = torch.randint(1, 100, (2048,), dtype=torch.int32).cuda()
    dst_hidden_state: torch.Tensor = torch.rand((src_hidden_state_size, 7168), dtype=torch.float16).cuda()
    dst_positions: torch.Tensor = torch.randint(1, 100, (1024,), dtype=torch.int64).cuda()
    dst_meta: torch.Tensor = torch.zeros(8, dtype=torch.uint16).cuda()

    buf: torch.Tensor = torch.zeros((128,7168*2), dtype=torch.uint8).cuda()


    for _ in  range(10):
        invoke_serialize_context_kernel(True, src_slot_mapping, src_context_lens, src_block_tables, src_hidden_state, src_positions, buf)

        # print("src_slot_mapping=", src_slot_mapping)
        # print("buf=", buf)

        invoke_deserialize_context_kernel(buf, dst_meta, dst_slot_mapping, dst_context_lens, dst_block_tables, dst_hidden_state, dst_positions)

    # print("dst_slot_mapping=", dst_slot_mapping)

    assert dst_meta[0] == 1
    assert dst_meta[1] == src_slot_mapping_size
    assert dst_meta[2] == src_context_lens_size
    assert dst_meta[3] == src_block_tables_size
    assert dst_meta[4] == src_hidden_state_size
    assert dst_meta[5] == src_positions_size
    

    assert torch.equal(src_slot_mapping, dst_slot_mapping[:src_slot_mapping_size])
    assert torch.equal(src_context_lens, dst_context_lens[:src_context_lens_size])
    assert torch.equal(src_block_tables, dst_block_tables[:src_block_tables_size])
    assert torch.equal(src_positions, dst_positions[:src_positions_size])

    assert torch.equal(src_hidden_state, dst_hidden_state)


if __name__ == "__main__":

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=30),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        for _ in range(5):
            test()
            prof.step()

    prof.export_chrome_trace(f"tracing-serdes-test.json.gz")
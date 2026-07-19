#!/usr/bin/env python3
"""Narrow smoke checks for the current V1 backport flags.

Run this with the same Python environment that imports the patched vLLM 0.24
package. The checks intentionally avoid full engine/bootstrap overhead and
focus only on the specific control-flow changes we introduced.
"""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch


VOCAB_SIZE = 1024


def _make_input_batch(*, device: torch.device, max_num_reqs: int = 2):
    from vllm.v1.worker.gpu_input_batch import InputBatch

    return InputBatch(
        max_num_reqs=max_num_reqs,
        max_model_len=64,
        max_num_batched_tokens=64,
        device=device,
        vocab_size=VOCAB_SIZE,
        block_sizes=[1],
        kernel_block_sizes=[1],
    )


def _make_request(
    req_id: str,
    *,
    allowed_token_ids: list[int] | None = None,
    output_token_ids: list[int] | None = None,
    presence_penalty: float = 0.0,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 32,
):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

    output_ids = [] if output_token_ids is None else list(output_token_ids)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        allowed_token_ids=allowed_token_ids,
        presence_penalty=presence_penalty,
    )
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=[],
        block_ids=([],),
        generator=None,
        num_computed_tokens=len([token for token in output_ids if token != -1]),
        output_token_ids=output_ids,
    )


def check_stale_allowed_mask(*, device: torch.device) -> None:
    os.environ["VLLM_V1_BACKPORT_CLEAR_STALE_ALLOWED_MASK"] = "1"

    batch = _make_input_batch(device=device, max_num_reqs=3)
    req0 = _make_request("req0", allowed_token_ids=[7, 8, 9])
    row0 = batch.add_request(req0)
    assert row0 == 0
    assert batch.allowed_token_ids_mask_cpu_tensor is not None

    batch.remove_request("req0")
    batch.allowed_token_ids_mask_cpu_tensor[row0].fill_(True)

    req1 = _make_request("req1")
    row1 = batch.add_request(req1)
    assert row1 == row0
    assert not batch.allowed_token_ids_mask_cpu_tensor[row1].any().item(), (
        "stale allowed-token row was not cleared"
    )

    print("PASS stale_allowed_mask_clear")


def check_async_output_repair(*, device: torch.device) -> None:
    os.environ["VLLM_V1_BACKPORT_ASYNC_OUTPUT_REPAIR_OPT"] = "1"

    batch = _make_input_batch(device=device)
    req_a = _make_request(
        "req_a",
        output_token_ids=[11, -1, -1],
        presence_penalty=0.1,
    )
    req_b = _make_request(
        "req_b",
        output_token_ids=[22],
        presence_penalty=0.1,
    )

    idx_a = batch.add_request(req_a)
    idx_b = batch.add_request(req_b)
    batch.refresh_metadata()
    batch.prev_req_id_to_index = {"req_a": idx_a, "req_b": idx_b}

    async_copy_ready_event = Mock()
    sampled_token_ids = torch.tensor(
        [[101, 102, -1], [202, 203, -1]],
        dtype=torch.int32,
    )
    batch.set_async_sampled_token_ids(sampled_token_ids, async_copy_ready_event)
    batch.update_async_output_token_ids()

    output_token_ids = batch.sampling_metadata.output_token_ids
    assert output_token_ids[idx_a] == [11, 101, 102], output_token_ids[idx_a]
    assert output_token_ids[idx_b] == [22], output_token_ids[idx_b]
    async_copy_ready_event.synchronize.assert_called_once()

    print("PASS async_output_repair_opt")


def _make_scheduler_output():
    scheduled_cached_reqs = SimpleNamespace(
        resumed_req_ids=set(),
        req_ids=[],
        num_computed_tokens=[],
        new_block_ids=[],
        num_output_tokens=[],
        new_token_ids=[],
        all_token_ids={},
    )
    return SimpleNamespace(
        finished_req_ids=[],
        new_block_ids_to_zero=[],
        free_encoder_mm_hashes=[],
        num_scheduled_tokens={},
        scheduled_cached_reqs=scheduled_cached_reqs,
        scheduled_spec_decode_tokens={},
        scheduled_new_reqs=[],
    )


def _make_runner(*, skip_noop_refresh: bool):
    batch_update_builder = SimpleNamespace(
        batch_changed=False,
        added=[],
        moved=[],
        has_removed=lambda: False,
    )
    input_batch = SimpleNamespace(
        req_id_to_index={},
        batch_update_builder=batch_update_builder,
        remove_request=Mock(),
        refresh_metadata=Mock(),
        add_request=Mock(),
        update_req_spec_token_ids=Mock(),
        condense=Mock(),
    )
    return SimpleNamespace(
        requests={},
        num_prompt_logprobs={},
        model_config=SimpleNamespace(logits_processors=None),
        vllm_config=SimpleNamespace(reasoning_config=None),
        is_pooling_model=False,
        num_spec_tokens=0,
        use_async_spec_decode=False,
        late_interaction_runner=SimpleNamespace(on_requests_finished=Mock()),
        input_batch=input_batch,
        encoder_cache={},
        speculative_config=None,
        _backport_debug_enabled=False,
        _backport_skip_noop_metadata_refresh=skip_noop_refresh,
        _backport_resident_unscheduled_batch=False,
        _record_backport_debug_stat=lambda *args, **kwargs: None,
        _zero_block_ids=Mock(),
        _may_reorder_batch=Mock(),
        _can_use_resident_unscheduled_batch=lambda scheduler_output: False,
    )


def check_runner_refresh_behavior() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    import vllm.v1.worker.gpu_model_runner as gpu_model_runner

    original_perf_counter = gpu_model_runner.time.perf_counter
    original_get_pp_group = gpu_model_runner.get_pp_group
    gpu_model_runner.get_pp_group = lambda: SimpleNamespace(is_last_rank=True)

    try:
        runner_skip = _make_runner(skip_noop_refresh=True)
        gpu_model_runner.time.perf_counter = lambda: (_ for _ in ()).throw(
            RuntimeError("perf_counter should not be called")
        )
        result = GPUModelRunner._update_states(runner_skip, _make_scheduler_output())
        assert result is None
        runner_skip.input_batch.refresh_metadata.assert_not_called()
        print("PASS skip_noop_metadata_refresh")

        runner_always = _make_runner(skip_noop_refresh=False)
        result = GPUModelRunner._update_states(
            runner_always, _make_scheduler_output()
        )
        assert result is None
        runner_always.input_batch.refresh_metadata.assert_called_once()
        print("PASS always_refresh_when_flag_off")

        print("PASS no_perf_counter_when_debug_off")
    finally:
        gpu_model_runner.time.perf_counter = original_perf_counter
        gpu_model_runner.get_pp_group = original_get_pp_group




def check_reorder_only_metadata_skip(*, device: torch.device) -> None:
    os.environ["VLLM_V1_BACKPORT_SKIP_REORDER_ONLY_METADATA_REBUILD"] = "1"

    greedy_batch = _make_input_batch(device=device)
    greedy_batch.add_request(
        _make_request("req0", temperature=0.0, top_p=1.0, top_k=0)
    )
    greedy_batch.add_request(
        _make_request("req1", temperature=0.0, top_p=1.0, top_k=0)
    )
    greedy_batch.refresh_metadata()
    greedy_before = greedy_batch.sampling_metadata
    greedy_batch.swap_states(0, 1)
    greedy_batch.refresh_metadata()
    assert greedy_batch.sampling_metadata is greedy_before

    sampled_batch = _make_input_batch(device=device)
    sampled_batch.add_request(
        _make_request("req0", temperature=0.7, top_p=0.91, top_k=17)
    )
    sampled_batch.add_request(
        _make_request("req1", temperature=1.3, top_p=0.73, top_k=9)
    )
    sampled_batch.refresh_metadata()
    sampled_before = sampled_batch.sampling_metadata
    sampled_batch.swap_states(0, 1)
    sampled_batch.refresh_metadata()
    assert sampled_batch.sampling_metadata is not sampled_before
    assert sampled_batch.sampling_metadata.temperature is not None
    assert sampled_batch.sampling_metadata.top_p is not None
    assert sampled_batch.sampling_metadata.top_k is not None
    assert torch.allclose(
        sampled_batch.sampling_metadata.temperature[:2].cpu(),
        torch.tensor([1.3, 0.7], dtype=torch.float32),
    )
    assert torch.allclose(
        sampled_batch.sampling_metadata.top_p[:2].cpu(),
        torch.tensor([0.73, 0.91], dtype=torch.float32),
    )
    assert torch.equal(
        sampled_batch.sampling_metadata.top_k[:2].cpu(),
        torch.tensor([9, 17], dtype=torch.int32),
    )

    batch_penalty = _make_input_batch(device=device)
    batch_penalty.add_request(
        _make_request("req0", temperature=0.0, top_p=1.0, top_k=0, presence_penalty=0.1)
    )
    batch_penalty.add_request(_make_request("req1", temperature=0.0, top_p=1.0, top_k=0))
    batch_penalty.refresh_metadata()
    penalty_before = batch_penalty.sampling_metadata
    batch_penalty.swap_states(0, 1)
    batch_penalty.refresh_metadata()
    assert batch_penalty.sampling_metadata is not penalty_before

    print("PASS reorder_only_metadata_skip")


def check_resident_prefix_view(*, device: torch.device) -> None:
    os.environ["VLLM_V1_BACKPORT_RESIDENT_UNSCHEDULED_BATCH"] = "1"

    batch = _make_input_batch(device=device, max_num_reqs=4)
    batch.add_request(
        _make_request("req0", output_token_ids=[10], presence_penalty=0.1)
    )
    batch.add_request(_make_request("req1", output_token_ids=[20, 21]))
    batch.add_request(
        _make_request("req2", output_token_ids=[30], presence_penalty=0.1)
    )

    assert batch.resident_num_reqs == 3
    assert batch.num_reqs == 3
    assert batch.req_ids == ["req0", "req1", "req2"]

    batch.activate_scheduled_prefix(["req2", "req0"])
    assert batch.resident_num_reqs == 3
    assert batch.num_reqs == 2
    assert batch.req_ids == ["req2", "req0"]
    assert batch.req_id_to_index == {"req0": 1, "req1": 2, "req2": 0}

    batch.refresh_metadata(force_rebuild=True)
    assert len(batch.sampling_metadata.output_token_ids) == 2
    assert batch.sampling_metadata.output_token_ids[0] == [30]
    assert batch.sampling_metadata.output_token_ids[1] == [10]
    assert len(batch.sampling_metadata.spec_token_ids or []) == 2

    print("PASS resident_prefix_view")


def check_runner_resident_unscheduled() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    import vllm.v1.worker.gpu_model_runner as gpu_model_runner

    batch_update_builder = SimpleNamespace(
        batch_changed=False,
        added=[],
        moved=[],
        has_removed=lambda: False,
    )
    input_batch = SimpleNamespace(
        req_id_to_index={"req0": 0, "req1": 1},
        logitsprocs_need_output_token_ids=False,
        batch_update_builder=batch_update_builder,
        remove_request=Mock(),
        refresh_metadata=Mock(),
        add_request=Mock(),
        update_req_spec_token_ids=Mock(),
        condense=Mock(),
        activate_scheduled_prefix=Mock(),
    )
    runner = SimpleNamespace(
        requests={
            "req0": _make_request("req0"),
            "req1": _make_request("req1"),
        },
        num_prompt_logprobs={},
        model_config=SimpleNamespace(logits_processors=None),
        vllm_config=SimpleNamespace(reasoning_config=None),
        is_pooling_model=False,
        num_spec_tokens=0,
        use_async_spec_decode=False,
        late_interaction_runner=SimpleNamespace(on_requests_finished=Mock()),
        input_batch=input_batch,
        encoder_cache={},
        speculative_config=None,
        _backport_debug_enabled=False,
        _backport_skip_noop_metadata_refresh=True,
        _backport_resident_unscheduled_batch=True,
        _record_backport_debug_stat=lambda *args, **kwargs: None,
        _zero_block_ids=Mock(),
        _may_reorder_batch=Mock(),
        _can_use_resident_unscheduled_batch=GPUModelRunner._can_use_resident_unscheduled_batch,
    )
    runner._can_use_resident_unscheduled_batch = (
        lambda scheduler_output: GPUModelRunner._can_use_resident_unscheduled_batch(
            runner, scheduler_output
        )
    )

    scheduler_output = _make_scheduler_output()
    scheduler_output.num_scheduled_tokens = {"req0": 1}

    original_get_pp_group = gpu_model_runner.get_pp_group
    gpu_model_runner.get_pp_group = lambda: SimpleNamespace(is_last_rank=True)
    try:
        result = GPUModelRunner._update_states(runner, scheduler_output)
        assert result is None
        input_batch.remove_request.assert_not_called()
        input_batch.activate_scheduled_prefix.assert_called_once_with(["req0"])
        input_batch.refresh_metadata.assert_called_once_with(force_rebuild=True)
    finally:
        gpu_model_runner.get_pp_group = original_get_pp_group

    print("PASS runner_resident_unscheduled")


def check_delta_block_table_commit(*, device: torch.device) -> None:
    from vllm.v1.worker.block_table import MultiGroupBlockTable

    block_table = MultiGroupBlockTable(
        max_num_reqs=4,
        max_model_len=32,
        max_num_batched_tokens=32,
        pin_memory=False,
        device=device,
        block_sizes=[1],
        kernel_block_sizes=[1],
        max_num_blocks=[8],
    )

    block_table.add_row(([11, 12],), 0)
    block_table.add_row(([21, 22],), 1)
    block_table.commit_block_table(2)
    gpu_before = block_table[0].get_device_tensor(2).cpu().clone()
    assert torch.equal(gpu_before[0, :2], torch.tensor([11, 12], dtype=torch.int32))
    assert torch.equal(gpu_before[1, :2], torch.tensor([21, 22], dtype=torch.int32))

    block_table.append_row(([13],), 0)
    block_table.clear_row(1)
    block_table.commit_block_table_delta(2)
    gpu_after = block_table[0].get_device_tensor(2).cpu()
    assert torch.equal(
        gpu_after[0, :3], torch.tensor([11, 12, 13], dtype=torch.int32)
    )
    assert torch.equal(gpu_after[1, :2], torch.tensor([0, 0], dtype=torch.int32))

    print("PASS delta_block_table_commit")


def check_runner_delta_block_table_flag() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    block_table = SimpleNamespace(
        commit_block_table=Mock(),
        commit_block_table_delta=Mock(),
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=1,
            block_table=block_table,
        ),
        _backport_delta_block_table_writes=True,
        arange_np=torch.arange(8, dtype=torch.int32).numpy(),
        query_pos=SimpleNamespace(np=torch.zeros(8, dtype=torch.int32).numpy()),
        _get_cumsum_and_arange=lambda num_scheduled_tokens, out: torch.tensor(
            [int(num_scheduled_tokens[0])], dtype=torch.int32
        ).numpy(),
    )
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    num_scheduled_tokens = torch.tensor([1], dtype=torch.int32).numpy()
    try:
        GPUModelRunner._prepare_inputs(runner, scheduler_output, num_scheduled_tokens)
    except AttributeError:
        # We only care that the block-table path was selected before deeper
        # tensor prep touches omitted mock attributes.
        pass

    block_table.commit_block_table_delta.assert_called_once_with(1)
    block_table.commit_block_table.assert_not_called()
    print("PASS runner_delta_block_table_flag")




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device used for InputBatch smoke checks.",
    )
    parser.add_argument(
        "--check",
        action="append",
        choices=(
            "stale-allowed-mask",
            "async-output-repair",
            "runner-refresh",
            "reorder-only-metadata-skip",
            "resident-prefix-view",
            "runner-resident-unscheduled",
            "delta-block-table-commit",
            "runner-delta-block-table-flag",
            "all",
        ),
        help="Specific smoke checks to run. Defaults to all.",
    )
    args = parser.parse_args()

    selected_checks = args.check or ["all"]
    if "all" in selected_checks:
        selected_checks = [
            "stale-allowed-mask",
            "async-output-repair",
            "runner-refresh",
            "reorder-only-metadata-skip",
            "resident-prefix-view",
            "runner-resident-unscheduled",
            "delta-block-table-commit",
            "runner-delta-block-table-flag",
        ]

    device = torch.device(args.device)

    if "stale-allowed-mask" in selected_checks:
        check_stale_allowed_mask(device=device)
    if "async-output-repair" in selected_checks:
        check_async_output_repair(device=device)
    if "runner-refresh" in selected_checks:
        check_runner_refresh_behavior()
    if "reorder-only-metadata-skip" in selected_checks:
        check_reorder_only_metadata_skip(device=device)
    if "resident-prefix-view" in selected_checks:
        check_resident_prefix_view(device=device)
    if "runner-resident-unscheduled" in selected_checks:
        check_runner_resident_unscheduled()
    if "delta-block-table-commit" in selected_checks:
        check_delta_block_table_commit(device=device)
    if "runner-delta-block-table-flag" in selected_checks:
        check_runner_delta_block_table_flag()


if __name__ == "__main__":
    main()

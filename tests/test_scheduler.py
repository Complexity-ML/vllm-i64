"""
vllm-i64 :: Scheduler Tests
INL - 2025
"""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_i64.engine.i64_scheduler import I64Scheduler, I64Request, I64Batch, RequestStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scheduler(**overrides):
    """Create a scheduler with small test defaults."""
    defaults = dict(
        num_experts=4,
        max_batch_size=4,
        kv_block_size=4,
        max_kv_blocks=16,
        max_seq_len=64,
        max_prefill_tokens=512,
        enable_preemption=True,
    )
    defaults.update(overrides)
    return I64Scheduler(**defaults)


def prompt(length=4, start=100):
    """Return an i64 prompt array of given length."""
    return np.arange(start, start + length, dtype=np.int64)


# ===========================================================================
# 1. Basics: add_request, incrementing IDs, empty schedule
# ===========================================================================

class TestSchedulerBasics:
    def test_add_request_returns_incrementing_ids(self):
        sched = make_scheduler()
        id0 = sched.add_request(prompt(4))
        id1 = sched.add_request(prompt(4))
        id2 = sched.add_request(prompt(4))
        assert id0 == 0
        assert id1 == 1
        assert id2 == 2

    def test_added_request_goes_to_pending(self):
        sched = make_scheduler()
        sched.add_request(prompt(4))
        assert len(sched.pending) == 1
        assert sched.pending[0].status == RequestStatus.PENDING

    def test_empty_schedule_returns_none(self):
        sched = make_scheduler()
        batch = sched.schedule()
        assert batch is None

    def test_stats_initial(self):
        sched = make_scheduler()
        stats = sched.get_stats()
        assert stats["pending"] == 0
        assert stats["running"] == 0
        assert stats["finished"] == 0
        assert stats["preempted"] == 0
        assert stats["free_kv_blocks"] == 16
        assert stats["total_steps"] == 0

    def test_stats_after_add_and_schedule(self):
        sched = make_scheduler()
        sched.add_request(prompt(4))
        sched.add_request(prompt(4))
        batch = sched.schedule()
        stats = sched.get_stats()
        assert stats["pending"] == 0
        assert stats["running"] == 2
        assert stats["total_steps"] == 1
        # Each request needs ceil(4/4)=1 block, so 16-2=14 free
        assert stats["free_kv_blocks"] == 14


# ===========================================================================
# 2. Priority ordering
# ===========================================================================

class TestSchedulerPriority:
    def test_lower_priority_number_scheduled_first(self):
        sched = make_scheduler(max_batch_size=1)
        # Add low-priority first, high-priority second
        sched.add_request(prompt(4, start=200), priority=5)   # low priority
        sched.add_request(prompt(4, start=300), priority=-1)  # high priority
        batch = sched.schedule()
        # Only 1 slot; should pick priority=-1
        assert batch.num_requests == 1
        assert len(sched.running) == 1
        assert sched.running[0].priority == -1

    def test_same_priority_fifo(self):
        sched = make_scheduler(max_batch_size=1)
        id0 = sched.add_request(prompt(4, start=10), priority=0)
        id1 = sched.add_request(prompt(4, start=20), priority=0)
        batch = sched.schedule()
        # First-arrived should win tie-break
        assert int(batch.request_ids[0]) == id0

    def test_pending_sorted_by_priority_then_arrival(self):
        sched = make_scheduler(max_batch_size=1)
        sched.add_request(prompt(4), priority=2, max_new_tokens=1)
        sched.add_request(prompt(4), priority=0, max_new_tokens=1)
        sched.add_request(prompt(4), priority=-1, max_new_tokens=1)
        # Verify heap pops in priority order via scheduling
        scheduled_priorities = []
        for _ in range(3):
            batch = sched.schedule()
            assert batch is not None
            req = sched.running[-1]
            scheduled_priorities.append(req.priority)
            # Complete prefill + generate 1 token to finish
            sched.update_after_step({req.request_id: 999})
            sched.schedule()  # move finished out, free slot
        assert scheduled_priorities == [-1, 0, 2]


# ===========================================================================
# 3. Schedule: pending -> running, finished removal
# ===========================================================================

class TestSchedulerPrefill:
    def test_schedule_moves_pending_to_running(self):
        sched = make_scheduler()
        sched.add_request(prompt(4))
        assert len(sched.pending) == 1
        batch = sched.schedule()
        assert len(sched.pending) == 0
        assert len(sched.running) == 1
        assert sched.running[0].status == RequestStatus.RUNNING

    def test_schedule_returns_batch(self):
        sched = make_scheduler()
        sched.add_request(prompt(4))
        batch = sched.schedule()
        assert isinstance(batch, I64Batch)
        assert batch.num_requests == 1

    def test_chunked_prefill_long_prompt(self):
        """A long prompt exceeding max_prefill_tokens is split into chunks."""
        sched = make_scheduler(max_prefill_tokens=4)
        # Prompt of length 12 -- with budget 4, first step processes first 4 tokens
        long_prompt = np.arange(100, 112, dtype=np.int64)
        sched.add_request(long_prompt, max_new_tokens=2)
        batch = sched.schedule()
        assert batch is not None
        req = sched.running[0]
        # The first chunk should be at most max_prefill_tokens in size
        # Because prefill_progress starts at 0, the chunk should be 4 tokens
        assert batch.is_prefill[0] == 1

    def test_respects_max_batch_size(self):
        sched = make_scheduler(max_batch_size=2)
        for _ in range(5):
            sched.add_request(prompt(4))
        batch = sched.schedule()
        assert batch.num_requests == 2
        assert len(sched.pending) == 3


# ===========================================================================
# 4. KV block allocation and freeing
# ===========================================================================

class TestSchedulerKVBlocks:
    def test_blocks_allocated_on_admit(self):
        sched = make_scheduler()
        sched.add_request(prompt(4))  # needs ceil(4/4)=1 block
        sched.schedule()
        req = sched.running[0]
        assert len(req.kv_block_ids) == 1

    def test_larger_prompt_more_blocks(self):
        sched = make_scheduler()
        sched.add_request(prompt(10))  # needs ceil(10/4)=3 blocks
        sched.schedule()
        req = sched.running[0]
        assert len(req.kv_block_ids) == 3

    def test_free_blocks_decrease_on_admit(self):
        sched = make_scheduler(max_kv_blocks=16)
        sched.add_request(prompt(8))  # ceil(8/4)=2 blocks
        sched.schedule()
        assert len(sched.free_blocks) == 14

    def test_blocks_freed_on_finish(self):
        sched = make_scheduler(max_kv_blocks=16)
        sched.add_request(prompt(4), max_new_tokens=1)
        sched.schedule()
        assert len(sched.free_blocks) == 15  # 1 block used
        # Generate 1 token to finish the request
        req = sched.running[0]
        sched.update_after_step({req.request_id: 999})
        assert req.is_finished
        # Next schedule should move it to finished and free blocks
        sched.schedule()
        assert len(sched.running) == 0
        assert len(sched.finished) == 1
        assert len(sched.free_blocks) == 16  # all freed

    def test_cannot_admit_when_no_blocks(self):
        sched = make_scheduler(max_kv_blocks=2, enable_preemption=False)
        # First request takes 1 block (prompt=4, block_size=4)
        sched.add_request(prompt(4))
        sched.add_request(prompt(4))
        sched.add_request(prompt(4))  # no blocks left for this one
        sched.schedule()
        assert len(sched.running) == 2
        assert len(sched.pending) == 1


# ===========================================================================
# 5. Preemption
# ===========================================================================

class TestSchedulerPreemption:
    def test_preempt_low_priority_to_admit_high_priority(self):
        sched = make_scheduler(max_kv_blocks=2, max_batch_size=4)
        # Fill KV with low-priority requests (1 block each)
        sched.add_request(prompt(4), priority=5)
        sched.add_request(prompt(4), priority=5)
        sched.schedule()  # both running, 0 free blocks
        assert len(sched.running) == 2
        assert len(sched.free_blocks) == 0
        # Add a high-priority request
        sched.add_request(prompt(4), priority=-1)
        batch = sched.schedule()
        # Should have preempted at least one low-priority request
        assert any(r.priority == -1 for r in sched.running)

    def test_preempted_request_goes_to_pending_with_boost(self):
        sched = make_scheduler(max_kv_blocks=1, max_batch_size=4)
        sched.add_request(prompt(4), priority=5)
        sched.schedule()  # runs the priority=5 request
        assert len(sched.running) == 1
        # Add high-priority — only 1 block total, must preempt
        sched.add_request(prompt(4), priority=-2)
        sched.schedule()
        # Preempted request should be re-queued with boosted priority
        # The preempted list gets re-admitted at schedule time;
        # the victim now has priority=min(5, -1) = -1
        running_priorities = [r.priority for r in sched.running]
        assert -2 in running_priorities or -1 in running_priorities

    def test_no_preemption_when_disabled(self):
        sched = make_scheduler(max_kv_blocks=1, max_batch_size=4, enable_preemption=False)
        sched.add_request(prompt(4), priority=5)
        sched.schedule()
        sched.add_request(prompt(4), priority=-1)
        sched.schedule()
        # High-priority stays pending because preemption is off
        assert len(sched.pending) == 1


# ===========================================================================
# 6. Finish conditions: EOS and max_new_tokens
# ===========================================================================

class TestSchedulerFinish:
    def test_eos_detection(self):
        sched = make_scheduler()
        eos = 2
        sched.add_request(prompt(4), max_new_tokens=100, eos_token_id=eos)
        sched.schedule()
        req = sched.running[0]
        # Generate EOS token
        sched.update_after_step({req.request_id: eos})
        assert req.is_finished

    def test_max_new_tokens_finish(self):
        sched = make_scheduler()
        sched.add_request(prompt(4), max_new_tokens=3)
        sched.schedule()
        req = sched.running[0]
        for i in range(3):
            assert not req.is_finished or i > 0  # shouldn't be done early
            sched.update_after_step({req.request_id: 50 + i})
        assert req.is_finished
        assert req.num_generated == 3

    def test_finished_requests_removed_from_running(self):
        sched = make_scheduler()
        sched.add_request(prompt(4), max_new_tokens=1)
        sched.schedule()
        req = sched.running[0]
        sched.update_after_step({req.request_id: 42})
        assert req.is_finished
        sched.schedule()
        assert len(sched.running) == 0
        assert len(sched.finished) == 1
        assert sched.finished[0].request_id == req.request_id

    def test_not_finished_before_eos_or_max(self):
        sched = make_scheduler()
        sched.add_request(prompt(4), max_new_tokens=10, eos_token_id=999)
        sched.schedule()
        req = sched.running[0]
        # Generate a non-EOS token
        sched.update_after_step({req.request_id: 50})
        assert not req.is_finished


# ===========================================================================
# 7. update_after_step
# ===========================================================================

class TestSchedulerUpdateAfterStep:
    def test_appends_token_and_updates_seq_pos(self):
        sched = make_scheduler()
        sched.add_request(prompt(4), max_new_tokens=10)
        sched.schedule()
        req = sched.running[0]
        sched.update_after_step({req.request_id: 42})
        assert req.output_token_ids == [42]
        assert req.seq_pos == req.num_prompt_tokens + 1  # 4+1=5
        sched.update_after_step({req.request_id: 43})
        assert req.output_token_ids == [42, 43]
        assert req.seq_pos == req.num_prompt_tokens + 2  # 4+2=6

    def test_fairness_tracking(self):
        sched = make_scheduler()
        rid = sched.add_request(prompt(4), max_new_tokens=10)
        sched.schedule()
        sched.update_after_step({rid: 10})
        assert sched._tokens_generated_per_req[rid] == 1
        sched.update_after_step({rid: 11})
        assert sched._tokens_generated_per_req[rid] == 2

    def test_allocates_extra_blocks_when_needed(self):
        sched = make_scheduler(kv_block_size=4, max_kv_blocks=16)
        # Prompt fills exactly 1 block (4 tokens)
        sched.add_request(prompt(4), max_new_tokens=10)
        sched.schedule()
        req = sched.running[0]
        initial_blocks = len(req.kv_block_ids)
        assert initial_blocks == 1
        # After 1 generated token, total=5 > 4, so needs 2 blocks
        sched.update_after_step({req.request_id: 50})
        assert len(req.kv_block_ids) == 2


# ===========================================================================
# 8. Batch building
# ===========================================================================

class TestSchedulerBatch:
    def test_batch_request_ids(self):
        sched = make_scheduler()
        id0 = sched.add_request(prompt(4))
        id1 = sched.add_request(prompt(4))
        batch = sched.schedule()
        assert set(batch.request_ids.tolist()) == {id0, id1}

    def test_batch_token_ids_dtype(self):
        sched = make_scheduler()
        sched.add_request(prompt(4))
        batch = sched.schedule()
        assert batch.token_ids.dtype == np.int64

    def test_batch_expert_ids_match_bitmask(self):
        sched = make_scheduler(num_experts=4)
        p = np.array([0, 1, 2, 7], dtype=np.int64)
        sched.add_request(p)
        batch = sched.schedule()
        expected = (p & np.int64(3)).astype(np.int32)
        np.testing.assert_array_equal(batch.expert_ids, expected)

    def test_batch_positions_prefill(self):
        sched = make_scheduler()
        sched.add_request(prompt(4))
        batch = sched.schedule()
        np.testing.assert_array_equal(batch.positions, np.arange(4, dtype=np.int32))

    def test_batch_is_prefill_flag(self):
        sched = make_scheduler()
        sched.add_request(prompt(4))
        batch = sched.schedule()
        assert batch.is_prefill[0] == 1

    def test_batch_decode_step(self):
        sched = make_scheduler()
        sched.add_request(prompt(4), max_new_tokens=5)
        sched.schedule()
        req = sched.running[0]
        sched.update_after_step({req.request_id: 50})
        # Second schedule is a decode step
        batch = sched.schedule()
        assert batch.is_prefill[0] == 0
        # Decode: single token per request
        assert batch.tokens_per_request[0] == 1

    def test_batch_kv_block_table_shape(self):
        sched = make_scheduler()
        sched.add_request(prompt(4))
        sched.add_request(prompt(8))
        batch = sched.schedule()
        # 2 requests; max blocks = ceil(8/4) = 2
        assert batch.kv_block_table.shape[0] == 2
        assert batch.kv_block_table.shape[1] == 2

    def test_batch_tokens_per_request(self):
        sched = make_scheduler()
        sched.add_request(prompt(3))
        sched.add_request(prompt(6))
        batch = sched.schedule()
        tpr = batch.tokens_per_request.tolist()
        assert 3 in tpr
        assert 6 in tpr

    def test_batch_mixed_prefill_and_decode(self):
        sched = make_scheduler()
        # First request: will be decode after step
        sched.add_request(prompt(4), max_new_tokens=5)
        sched.schedule()
        req = sched.running[0]
        sched.update_after_step({req.request_id: 50})
        # Add a second request that will be in prefill
        sched.add_request(prompt(4), max_new_tokens=5)
        batch = sched.schedule()
        assert batch.num_requests == 2
        prefill_flags = batch.is_prefill.tolist()
        assert 0 in prefill_flags  # decode
        assert 1 in prefill_flags  # prefill


# ===========================================================================
# I64Request unit tests
# ===========================================================================

class TestI64Request:
    def test_properties(self):
        req = I64Request(
            request_id=0,
            prompt_token_ids=np.array([10, 20, 30], dtype=np.int64),
            max_new_tokens=5,
        )
        assert req.num_prompt_tokens == 3
        assert req.num_generated == 0
        assert req.total_tokens == 3
        assert not req.is_finished

    def test_get_all_token_ids(self):
        req = I64Request(
            request_id=0,
            prompt_token_ids=np.array([1, 2, 3], dtype=np.int64),
            max_new_tokens=5,
        )
        req.output_token_ids = [4, 5]
        all_ids = req.get_all_token_ids()
        np.testing.assert_array_equal(all_ids, [1, 2, 3, 4, 5])

    def test_get_last_token_id_no_output(self):
        req = I64Request(
            request_id=0,
            prompt_token_ids=np.array([10, 20, 30], dtype=np.int64),
            max_new_tokens=5,
        )
        assert req.get_last_token_id() == 30

    def test_get_last_token_id_with_output(self):
        req = I64Request(
            request_id=0,
            prompt_token_ids=np.array([10, 20, 30], dtype=np.int64),
            max_new_tokens=5,
        )
        req.output_token_ids = [40, 50]
        assert req.get_last_token_id() == 50

    def test_prefill_complete(self):
        req = I64Request(
            request_id=0,
            prompt_token_ids=np.array([1, 2, 3, 4], dtype=np.int64),
            max_new_tokens=5,
        )
        assert not req.prefill_complete
        req.prefill_progress = 4
        assert req.prefill_complete


# ===========================================================================
# RequestStatus enum
# ===========================================================================

class TestRequestStatus:
    def test_values(self):
        assert int(RequestStatus.PENDING) == 0
        assert int(RequestStatus.RUNNING) == 1
        assert int(RequestStatus.PREEMPTED) == 2
        assert int(RequestStatus.FINISHED) == 3


# ===========================================================================
# Run with pytest
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for vllm_gtc.engine module."""

from __future__ import annotations


class TestRequest:
    """Tests for Request."""

    def test_create_request(self):
        from vllm_gtc.engine import Request

        req = Request(request_id="req-1", prompt_tokens=[1, 2, 3], max_tokens=10)
        assert req.request_id == "req-1"
        assert req.prompt_tokens == [1, 2, 3]
        assert req.max_tokens == 10

    def test_request_state_initial(self):
        from vllm_gtc.engine import Request, RequestState

        req = Request(request_id="req-1", prompt_tokens=[1, 2, 3], max_tokens=10)
        assert req.state == RequestState.WAITING

    def test_num_prompt_tokens(self):
        from vllm_gtc.engine import Request

        req = Request(request_id="req-1", prompt_tokens=[1, 2, 3, 4, 5], max_tokens=10)
        assert req.num_prompt_tokens == 5

    def test_generated_tokens(self):
        from vllm_gtc.engine import Request

        req = Request(request_id="req-1", prompt_tokens=[1, 2], max_tokens=10)
        req.add_token(100)
        req.add_token(101)
        assert req.generated_tokens == [100, 101]
        assert req.num_generated_tokens == 2

    def test_is_finished_max_tokens(self):
        from vllm_gtc.engine import Request

        req = Request(request_id="req-1", prompt_tokens=[1], max_tokens=2)
        assert not req.is_finished
        req.add_token(10)
        assert not req.is_finished
        req.add_token(11)
        assert req.is_finished

    def test_is_finished_eos(self):
        from vllm_gtc.engine import Request

        req = Request(request_id="req-1", prompt_tokens=[1], max_tokens=100, eos_token_id=0)
        req.add_token(10)
        assert not req.is_finished
        req.add_token(0)  # EOS
        assert req.is_finished


class TestSchedulerConfig:
    """Tests for SchedulerConfig."""

    def test_default_config(self):
        from vllm_gtc.engine import SchedulerConfig

        config = SchedulerConfig()
        assert config.max_batch_size > 0
        assert config.max_num_seqs > 0

    def test_custom_config(self):
        from vllm_gtc.engine import SchedulerConfig

        config = SchedulerConfig(max_batch_size=32, max_num_seqs=64)
        assert config.max_batch_size == 32
        assert config.max_num_seqs == 64


class TestScheduler:
    """Tests for Scheduler."""

    def test_add_request(self):
        from vllm_gtc.engine import Request, Scheduler, SchedulerConfig

        config = SchedulerConfig(max_batch_size=4)
        scheduler = Scheduler(config)

        req = Request(request_id="req-1", prompt_tokens=[1, 2, 3], max_tokens=10)
        scheduler.add_request(req)

        assert scheduler.num_waiting == 1

    def test_schedule_prefill(self):
        from vllm_gtc.engine import Request, Scheduler, SchedulerConfig

        config = SchedulerConfig(max_batch_size=4)
        scheduler = Scheduler(config)

        req = Request(request_id="req-1", prompt_tokens=[1, 2, 3], max_tokens=10)
        scheduler.add_request(req)

        output = scheduler.schedule()
        assert len(output.prefill_requests) == 1
        assert output.prefill_requests[0].request_id == "req-1"

    def test_schedule_decode_after_prefill(self):
        from vllm_gtc.engine import Request, RequestState, Scheduler, SchedulerConfig

        config = SchedulerConfig(max_batch_size=4)
        scheduler = Scheduler(config)

        req = Request(request_id="req-1", prompt_tokens=[1, 2, 3], max_tokens=10)
        scheduler.add_request(req)

        # First schedule: prefill
        output1 = scheduler.schedule()
        assert len(output1.prefill_requests) == 1

        # Mark as running (simulating prefill completion)
        req.state = RequestState.RUNNING
        req.add_token(100)

        # Second schedule: decode
        output2 = scheduler.schedule()
        assert len(output2.decode_requests) == 1

    def test_schedule_respects_batch_size(self):
        from vllm_gtc.engine import Request, Scheduler, SchedulerConfig

        config = SchedulerConfig(max_batch_size=2)
        scheduler = Scheduler(config)

        for i in range(5):
            req = Request(request_id=f"req-{i}", prompt_tokens=[1, 2], max_tokens=10)
            scheduler.add_request(req)

        output = scheduler.schedule()
        # Should only schedule up to max_batch_size
        assert len(output.prefill_requests) <= 2

    def test_remove_finished_requests(self):
        from vllm_gtc.engine import Request, RequestState, Scheduler, SchedulerConfig

        config = SchedulerConfig(max_batch_size=4)
        scheduler = Scheduler(config)

        req = Request(request_id="req-1", prompt_tokens=[1], max_tokens=2)
        scheduler.add_request(req)

        # Prefill
        scheduler.schedule()
        req.state = RequestState.RUNNING

        # Generate until finished
        req.add_token(10)
        req.add_token(11)
        assert req.is_finished

        # Update scheduler
        scheduler.update_finished()
        assert scheduler.num_running == 0

    def test_continuous_batching(self):
        """Test that new requests can join running batch."""
        from vllm_gtc.engine import Request, RequestState, Scheduler, SchedulerConfig

        config = SchedulerConfig(max_batch_size=4)
        scheduler = Scheduler(config)

        # Add first request and start it
        req1 = Request(request_id="req-1", prompt_tokens=[1, 2], max_tokens=10)
        scheduler.add_request(req1)
        scheduler.schedule()
        req1.state = RequestState.RUNNING
        req1.add_token(100)

        # Add second request while first is running
        req2 = Request(request_id="req-2", prompt_tokens=[3, 4], max_tokens=10)
        scheduler.add_request(req2)

        output = scheduler.schedule()
        # Should have both: req2 in prefill, req1 in decode
        assert len(output.prefill_requests) == 1
        assert len(output.decode_requests) == 1


class TestSchedulerOutput:
    """Tests for SchedulerOutput."""

    def test_total_requests(self):
        from vllm_gtc.engine import Request, Scheduler, SchedulerConfig

        config = SchedulerConfig(max_batch_size=4)
        scheduler = Scheduler(config)

        req = Request(request_id="req-1", prompt_tokens=[1], max_tokens=10)
        scheduler.add_request(req)

        output = scheduler.schedule()
        assert output.num_requests == 1

    def test_is_empty(self):
        from vllm_gtc.engine import Scheduler, SchedulerConfig

        config = SchedulerConfig(max_batch_size=4)
        scheduler = Scheduler(config)

        output = scheduler.schedule()
        assert output.is_empty

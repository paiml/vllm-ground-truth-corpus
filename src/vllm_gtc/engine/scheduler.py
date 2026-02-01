"""Continuous batching scheduler for LLM inference.

Implements the core scheduling logic from vLLM that enables
continuous batching - dynamically adding/removing requests
from a running batch.

References:
    - vLLM Scheduler: https://github.com/vllm-project/vllm
    - Orca: https://www.usenix.org/conference/osdi22/presentation/yu

Rust cross-reference:
    realizar::scheduler::RequestScheduler implements continuous
    batching with priority queues and memory-aware scheduling.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class RequestState(Enum):
    """State of a request in the scheduler."""

    WAITING = auto()  # In queue, not yet processed
    RUNNING = auto()  # Currently being processed
    FINISHED = auto()  # Completed or aborted


@dataclass
class Request:
    """A single inference request.

    Args:
        request_id: Unique identifier.
        prompt_tokens: Tokenized prompt.
        max_tokens: Maximum tokens to generate.
        eos_token_id: End-of-sequence token (None to disable).

    Examples:
        >>> req = Request("req-1", prompt_tokens=[1, 2, 3], max_tokens=100)
        >>> req.add_token(42)
        >>> req.generated_tokens
        [42]

    Rust equivalent:
        realizar::scheduler::Request

    """

    request_id: str
    prompt_tokens: list[int]
    max_tokens: int
    eos_token_id: int | None = None
    state: RequestState = RequestState.WAITING
    _generated: list[int] = field(default_factory=list, repr=False)

    @property
    def num_prompt_tokens(self) -> int:
        """Number of tokens in the prompt."""
        return len(self.prompt_tokens)

    @property
    def generated_tokens(self) -> list[int]:
        """List of generated tokens."""
        return list(self._generated)

    @property
    def num_generated_tokens(self) -> int:
        """Number of tokens generated so far."""
        return len(self._generated)

    def add_token(self, token: int) -> None:
        """Add a generated token."""
        self._generated.append(token)

    @property
    def is_finished(self) -> bool:
        """Check if request is complete."""
        if self.num_generated_tokens >= self.max_tokens:
            return True
        return bool(
            self.eos_token_id is not None
            and self._generated
            and self._generated[-1] == self.eos_token_id
        )


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler.

    Args:
        max_batch_size: Maximum requests in a single batch.
        max_num_seqs: Maximum concurrent sequences.

    Rust equivalent:
        realizar::scheduler::SchedulerConfig

    """

    max_batch_size: int = 256
    max_num_seqs: int = 256


@dataclass
class SchedulerOutput:
    """Output from a scheduling step.

    Args:
        prefill_requests: Requests needing prefill (prompt processing).
        decode_requests: Requests in decode phase (generation).

    Rust equivalent:
        realizar::scheduler::SchedulerOutput

    """

    prefill_requests: list[Request] = field(default_factory=list)
    decode_requests: list[Request] = field(default_factory=list)

    @property
    def num_requests(self) -> int:
        """Total number of scheduled requests."""
        return len(self.prefill_requests) + len(self.decode_requests)

    @property
    def is_empty(self) -> bool:
        """Check if no requests are scheduled."""
        return self.num_requests == 0


class Scheduler:
    """Continuous batching scheduler.

    Manages request queues and decides which requests to process
    in each iteration. Supports interleaving prefill (prompt
    processing) with decode (token generation).

    Args:
        config: Scheduler configuration.

    Examples:
        >>> config = SchedulerConfig(max_batch_size=32)
        >>> scheduler = Scheduler(config)
        >>> req = Request("req-1", [1, 2, 3], max_tokens=10)
        >>> scheduler.add_request(req)
        >>> output = scheduler.schedule()

    Rust equivalent:
        realizar::scheduler::RequestScheduler with Heijunka
        (load-leveling) principles from Toyota Production System.

    """

    def __init__(self, config: SchedulerConfig) -> None:
        """Initialize scheduler with configuration."""
        self.config = config
        self._waiting: list[Request] = []
        self._running: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a new request to the waiting queue.

        Args:
            request: Request to add.

        """
        request.state = RequestState.WAITING
        self._waiting.append(request)

    @property
    def num_waiting(self) -> int:
        """Number of requests waiting for prefill."""
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        """Number of requests in decode phase."""
        return len(self._running)

    def schedule(self) -> SchedulerOutput:
        """Determine which requests to process.

        Implements continuous batching by selecting:
        1. Running requests that need another decode step
        2. Waiting requests that can fit in remaining batch capacity

        Returns:
            SchedulerOutput with prefill and decode requests.

        """
        output = SchedulerOutput()

        # First, schedule running requests for decode
        budget = self.config.max_batch_size
        for req in self._running:
            if budget <= 0:
                break
            if not req.is_finished:
                output.decode_requests.append(req)
                budget -= 1

        # Then, schedule waiting requests for prefill
        new_running = []
        while self._waiting and budget > 0:
            req = self._waiting.pop(0)
            output.prefill_requests.append(req)
            new_running.append(req)
            budget -= 1

        # Move prefilled requests to running
        self._running.extend(new_running)

        return output

    def update_finished(self) -> list[Request]:
        """Remove finished requests from running queue.

        Returns:
            List of finished requests.

        """
        finished = [req for req in self._running if req.is_finished]
        self._running = [req for req in self._running if not req.is_finished]

        for req in finished:
            req.state = RequestState.FINISHED

        return finished

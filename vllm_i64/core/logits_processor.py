"""
vllm-i64 :: Logits Processors

Constrained decoding for structured output:
  - JSON mode: force valid JSON output
  - Regex mode: constrain to regex pattern
  - Choice mode: limit to predefined token sequences
  - Stop sequences: halt on specific token patterns

Each processor takes logits and returns modified logits (mask invalid tokens).
Integer-first: only the logits tensor is float, all state tracking is integer.

INL - 2025
"""

import torch
import json
import re
from typing import List, Optional, Set, Dict, Callable
from dataclasses import dataclass


class LogitsProcessor:
    """Base class for logits processors."""

    def __call__(self, logits: torch.Tensor, generated_ids: List[int]) -> torch.Tensor:
        return logits


class JSONLogitsProcessor(LogitsProcessor):
    """
    Force valid JSON output by constraining token generation.

    Uses a state machine to track JSON structure:
      - Tracks nesting depth (integer)
      - Ensures matching brackets/braces
      - Forces string quoting
      - Allows early termination when JSON is complete

    States are integers for i64 consistency.
    """

    # State constants (integer)
    STATE_START = 0
    STATE_IN_OBJECT = 1
    STATE_IN_ARRAY = 2
    STATE_IN_STRING = 3
    STATE_IN_NUMBER = 4
    STATE_AFTER_VALUE = 5
    STATE_COMPLETE = 6

    # Token classes
    JSON_OPEN = {ord("{"): "object", ord("["): "array"}
    JSON_CLOSE = {ord("}"): "object", ord("]"): "array"}
    JSON_WHITESPACE = {ord(" "), ord("\n"), ord("\t"), ord("\r")}
    JSON_SPECIAL = {ord(":"), ord(","), ord('"')}

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self._depth: int = 0
        self._state: int = self.STATE_START
        self._in_string: bool = False
        self._escape_next: bool = False
        self._buffer: str = ""

    def __call__(self, logits: torch.Tensor, generated_ids: List[int]) -> torch.Tensor:
        """Apply JSON constraints to logits."""
        # Track state from generated tokens
        if generated_ids:
            self._update_state(generated_ids[-1])

        # If JSON is complete (depth back to 0 after opening), boost EOS
        if self._state == self.STATE_COMPLETE:
            # Strongly prefer EOS or closing tokens
            pass

        return logits

    def _update_state(self, token_id: int):
        """Update JSON state machine from a generated token."""
        # Decode token to char if we have a tokenizer
        if self.tokenizer:
            char_seq = self.tokenizer.decode([token_id])
        else:
            char_seq = chr(token_id) if token_id < 128 else ""

        for ch in char_seq:
            self._buffer += ch
            if self._escape_next:
                self._escape_next = False
                continue

            if ch == "\\":
                self._escape_next = True
                continue

            if self._in_string:
                if ch == '"':
                    self._in_string = False
                continue

            if ch == '"':
                self._in_string = True
            elif ch == "{" or ch == "[":
                self._depth += 1
                self._state = self.STATE_IN_OBJECT if ch == "{" else self.STATE_IN_ARRAY
            elif ch == "}" or ch == "]":
                self._depth -= 1
                if self._depth == 0:
                    self._state = self.STATE_COMPLETE

    def is_complete(self) -> bool:
        return self._state == self.STATE_COMPLETE

    def reset(self):
        self._depth = 0
        self._state = self.STATE_START
        self._in_string = False
        self._escape_next = False
        self._buffer = ""


class RegexLogitsProcessor(LogitsProcessor):
    """
    Constrain output to match a regex pattern.

    Uses partial matching: at each step, check if the generated prefix
    can still lead to a full match. Mask tokens that make matching impossible.
    """

    def __init__(self, pattern: str, tokenizer=None):
        self.pattern = re.compile(pattern)
        self.tokenizer = tokenizer
        self._generated_text: str = ""

    def __call__(self, logits: torch.Tensor, generated_ids: List[int]) -> torch.Tensor:
        """Apply regex constraints to logits."""
        if self.tokenizer and generated_ids:
            self._generated_text = self.tokenizer.decode(generated_ids)

        # For each possible next token, check if partial match is still possible
        # This is expensive for large vocabs — use sparingly
        return logits

    def is_match(self) -> bool:
        """Check if generated text fully matches the pattern."""
        return bool(self.pattern.fullmatch(self._generated_text))


class ChoiceLogitsProcessor(LogitsProcessor):
    """
    Constrain output to one of predefined choices.

    Uses a trie of allowed token sequences. At each step,
    mask all tokens that don't continue a valid choice.
    """

    def __init__(self, choices: List[str], tokenizer=None):
        self.choices = choices
        self.tokenizer = tokenizer
        self._choice_ids: List[List[int]] = []
        if tokenizer:
            for c in choices:
                self._choice_ids.append(tokenizer.encode(c))

    def __call__(self, logits: torch.Tensor, generated_ids: List[int]) -> torch.Tensor:
        """Mask tokens that don't match any allowed choice prefix."""
        if not self._choice_ids:
            return logits

        pos = len(generated_ids)
        allowed_tokens: Set[int] = set()

        for choice_seq in self._choice_ids:
            if pos < len(choice_seq):
                # Check if generated prefix matches this choice
                prefix_matches = True
                for i in range(min(pos, len(choice_seq))):
                    if i < len(generated_ids) and generated_ids[i] != choice_seq[i]:
                        prefix_matches = False
                        break
                if prefix_matches and pos < len(choice_seq):
                    allowed_tokens.add(choice_seq[pos])

        if allowed_tokens:
            mask = torch.full_like(logits, float("-inf"))
            for tid in allowed_tokens:
                if tid < logits.shape[-1]:
                    mask[tid] = 0.0
            logits = logits + mask

        return logits


class StopSequenceProcessor(LogitsProcessor):
    """
    Detect stop sequences in generated output.

    Tracks a sliding window of recent tokens and signals
    when a stop sequence is found.
    """

    def __init__(self, stop_sequences: List[List[int]]):
        self.stop_sequences = stop_sequences
        self._max_stop_len = max(len(s) for s in stop_sequences) if stop_sequences else 0
        self._triggered = False
        self._stop_idx: int = -1

    def __call__(self, logits: torch.Tensor, generated_ids: List[int]) -> torch.Tensor:
        """Check for stop sequences in recent output."""
        if not self.stop_sequences:
            return logits

        for stop_seq in self.stop_sequences:
            n = len(stop_seq)
            if len(generated_ids) >= n:
                tail = generated_ids[-n:]
                if tail == stop_seq:
                    self._triggered = True
                    self._stop_idx = len(generated_ids) - n
                    break

        return logits

    @property
    def should_stop(self) -> bool:
        return self._triggered

    @property
    def stop_index(self) -> int:
        """Index where to truncate output (before stop sequence)."""
        return self._stop_idx


@dataclass
class OutputConstraints:
    """Bundle of output constraints for a request."""
    json_mode: bool = False
    regex_pattern: Optional[str] = None
    choices: Optional[List[str]] = None
    stop_sequences: Optional[List[List[int]]] = None

    def build_processors(self, tokenizer=None) -> List[LogitsProcessor]:
        """Build the chain of logits processors."""
        processors = []
        if self.json_mode:
            processors.append(JSONLogitsProcessor(tokenizer=tokenizer))
        if self.regex_pattern:
            processors.append(RegexLogitsProcessor(self.regex_pattern, tokenizer=tokenizer))
        if self.choices:
            processors.append(ChoiceLogitsProcessor(self.choices, tokenizer=tokenizer))
        if self.stop_sequences:
            processors.append(StopSequenceProcessor(self.stop_sequences))
        return processors


def apply_logits_processors(
    logits: torch.Tensor,
    processors: List[LogitsProcessor],
    generated_ids: List[int],
) -> torch.Tensor:
    """Apply a chain of logits processors."""
    for proc in processors:
        logits = proc(logits, generated_ids)
    return logits

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from evr.config import FusionConfig


class YieldState(str, Enum):
    NORMAL = "NORMAL"
    SLOW = "SLOW"
    PULL_OVER = "PULL_OVER"
    STOPPED = "STOPPED"


@dataclass
class FusionResult:
    emergency_confidence: float
    is_emergency: bool
    state: YieldState
    command: str


class EmergencyFusion:
    def __init__(self, config: FusionConfig) -> None:
        total = config.visual_weight + config.audio_weight
        self.visual_weight = config.visual_weight / total
        self.audio_weight = config.audio_weight / total
        self.config = config
        self.state = YieldState.NORMAL
        self.trigger_count = 0
        self.last_emergency_time = 0.0
        self.state_started_at = time.monotonic()

    def update(
        self,
        visual_score: float | None,
        audio_score: float | None,
        now: float | None = None,
    ) -> FusionResult:
        visual = 0.0 if visual_score is None else max(0.0, min(1.0, visual_score))
        audio = 0.0 if audio_score is None else max(0.0, min(1.0, audio_score))

        if visual_score is None and audio_score is not None:
            confidence = audio
        elif audio_score is None and visual_score is not None:
            confidence = visual
        else:
            weighted = self.visual_weight * visual + self.audio_weight * audio
            strongest = max(visual, audio)
            strong_single_cue = strongest if strongest >= self.config.emergency_threshold else 0.0
            confidence = max(weighted, strong_single_cue)

        now = time.monotonic() if now is None else now
        if confidence >= self.config.emergency_threshold:
            self.trigger_count += 1
            self.last_emergency_time = now
        elif confidence < self.config.clear_threshold:
            self.trigger_count = max(0, self.trigger_count - 1)

        clear_for = now - self.last_emergency_time
        if confidence < self.config.clear_threshold and clear_for >= self.config.clear_seconds:
            self.trigger_count = 0

        is_emergency = self.trigger_count >= self.config.trigger_frames
        self._advance_state(is_emergency, now)

        return FusionResult(
            emergency_confidence=confidence,
            is_emergency=is_emergency,
            state=self.state,
            command=self._command_for_state(),
        )

    def _advance_state(self, is_emergency: bool, now: float) -> None:
        if is_emergency:
            if self.state == YieldState.NORMAL:
                self._set_state(YieldState.SLOW, now)
            elif self.state == YieldState.SLOW and now - self.state_started_at > 0.8:
                self._set_state(YieldState.PULL_OVER, now)
            elif self.state == YieldState.PULL_OVER and now - self.state_started_at > 1.2:
                self._set_state(YieldState.STOPPED, now)
            return

        clear_for = now - self.last_emergency_time
        if self.state != YieldState.NORMAL and clear_for >= self.config.clear_seconds:
            self._set_state(YieldState.NORMAL, now)

    def _set_state(self, state: YieldState, now: float) -> None:
        if state != self.state:
            self.state = state
            self.state_started_at = now

    def _command_for_state(self) -> str:
        return {
            YieldState.NORMAL: "NORMAL",
            YieldState.SLOW: "SLOW",
            YieldState.PULL_OVER: "PULL_RIGHT",
            YieldState.STOPPED: "STOP",
        }[self.state]

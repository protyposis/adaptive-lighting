import asyncio
from typing import Any

from homeassistant.core import Event, State

from .utils import _AsyncSingleShotTimer


class Light:
    def __init__(self, entity_id: str):
        """Initialize the timer."""
        self.entity_id = entity_id

        # Tracks 'light.turn_off' service call
        self.turn_off_event: Event | None = None
        # Tracks 'light.turn_on' service call
        self.turn_on_event: Event | None = None
        # Keep 'asyncio.sleep' task that can be cancelled by 'light.turn_on' events
        self.sleep_task: asyncio.Task | None = None
        # Tracks if light is manually controlled
        self.manual_control: bool = False
        # Track 'state_changed' events of self.lights resulting from this integration
        self.last_state_change: list[State] = []
        # Track last 'service_data' to 'light.turn_on' resulting from this integration
        self.last_service_data: dict[str, Any] = {}
        # Track ongoing split adaptations to be able to cancel them
        self.split_adaptation_task: asyncio.Task | None = None

        # Track auto reset of manual_control
        self.auto_reset_manual_control_timer: _AsyncSingleShotTimer | None = None
        self.auto_reset_manual_control_time: float | None = None

        # Track light transition
        self.transition_timer: _AsyncSingleShotTimer | None = None

    def reset(self, reset_manual_control=True) -> None:
        """Reset the 'manual_control' status of the lights."""
        if reset_manual_control:
            self.manual_control = False
            timer = self.auto_reset_manual_control_timer
            if (timer := self.auto_reset_manual_control_timer) is not None:
                timer.cancel()
            self.auto_reset_manual_control_timer = None

        self.last_state_change = None
        self.last_service_data = None

    def process_turn_on_event(self, event: Event):
        if (task := self.sleep_task) is not None:
            task.cancel()

        self.turn_on_event = event

        if (
            (timer := self.auto_reset_manual_control_timer) is not None
            and timer.is_running()
            and event.time_fired > timer.start_time
        ):
            # Restart the auto reset timer
            timer.start()

    def process_turn_off_event(self, event: Event):
        self.turn_off_event = event
        self.reset()

        if (task := self.split_adaptation_task) is not None:
            task.cancel()

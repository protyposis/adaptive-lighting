"""Switch for the Adaptive Lighting integration."""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from copy import deepcopy
import datetime
from datetime import timedelta
import functools
import logging
from typing import Any

from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_BRIGHTNESS_PCT,
    ATTR_BRIGHTNESS_STEP,
    ATTR_BRIGHTNESS_STEP_PCT,
    ATTR_COLOR_NAME,
    ATTR_COLOR_TEMP_KELVIN,
    ATTR_HS_COLOR,
    ATTR_RGB_COLOR,
    ATTR_TRANSITION,
    ATTR_XY_COLOR,
)
from homeassistant.components.light import (
    is_on,
)
from homeassistant.components.light import DOMAIN as LIGHT_DOMAIN
from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN
from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_SERVICE,
    ATTR_SERVICE_DATA,
    CONF_NAME,
    EVENT_CALL_SERVICE,
    EVENT_HOMEASSISTANT_STARTED,
    EVENT_STATE_CHANGED,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_OFF,
    STATE_ON,
)
from homeassistant.core import (
    Context,
    Event,
    HomeAssistant,
    ServiceCall,
    State,
    callback,
)
from homeassistant.helpers import entity_platform, entity_registry
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.sun import get_astral_location
from homeassistant.util import slugify
import homeassistant.util.dt as dt_util
import voluptuous as vol

from .utils import (
    _split_service_data,
    create_context,
    get_entity_ids_from_service_event,
    is_light_on_off_event,
    is_our_context,
    _expand_light_groups,
    _attributes_have_changed,
    _AsyncSingleShotTimer,
    match_switch_state_event,
    _supported_features,
)
from .light import Light
from .sun import SunLightSettings

from .const import (
    ADAPT_BRIGHTNESS_SWITCH,
    ADAPT_COLOR_SWITCH,
    ATTR_ADAPT_BRIGHTNESS,
    ATTR_ADAPT_COLOR,
    ATTR_TURN_ON_OFF_LISTENER,
    CONF_ADAPT_DELAY,
    CONF_ADAPT_UNTIL_SLEEP,
    CONF_AUTORESET_CONTROL,
    CONF_DETECT_NON_HA_CHANGES,
    CONF_INCLUDE_CONFIG_IN_ATTRIBUTES,
    CONF_INITIAL_TRANSITION,
    CONF_INTERVAL,
    CONF_LIGHTS,
    CONF_MANUAL_CONTROL,
    CONF_MAX_BRIGHTNESS,
    CONF_MAX_COLOR_TEMP,
    CONF_MAX_SUNRISE_TIME,
    CONF_MIN_BRIGHTNESS,
    CONF_MIN_COLOR_TEMP,
    CONF_MIN_SUNSET_TIME,
    CONF_ONLY_ONCE,
    CONF_PREFER_RGB_COLOR,
    CONF_SEND_SPLIT_DELAY,
    CONF_SEPARATE_TURN_ON_COMMANDS,
    CONF_SLEEP_BRIGHTNESS,
    CONF_SLEEP_COLOR_TEMP,
    CONF_SLEEP_RGB_COLOR,
    CONF_SLEEP_RGB_OR_COLOR_TEMP,
    CONF_SLEEP_TRANSITION,
    CONF_SUNRISE_OFFSET,
    CONF_SUNRISE_TIME,
    CONF_SUNSET_OFFSET,
    CONF_SUNSET_TIME,
    CONF_TAKE_OVER_CONTROL,
    CONF_TRANSITION,
    CONF_TURN_ON_LIGHTS,
    CONF_USE_DEFAULTS,
    DOMAIN,
    EXTRA_VALIDATION,
    ICON_BRIGHTNESS,
    ICON_COLOR_TEMP,
    ICON_MAIN,
    ICON_SLEEP,
    SERVICE_APPLY,
    SERVICE_CHANGE_SWITCH_SETTINGS,
    SERVICE_SET_MANUAL_CONTROL,
    SET_MANUAL_CONTROL_SCHEMA,
    SLEEP_MODE_SWITCH,
    TURNING_OFF_DELAY,
    VALIDATION_TUPLES,
    apply_service_schema,
    replace_none_str,
)

_LOGGER = logging.getLogger(__name__)

SCAN_INTERVAL = timedelta(seconds=10)

COLOR_ATTRS = {  # Should ATTR_PROFILE be in here?
    ATTR_COLOR_NAME,
    ATTR_COLOR_TEMP_KELVIN,
    ATTR_HS_COLOR,
    ATTR_RGB_COLOR,
    ATTR_XY_COLOR,
}

BRIGHTNESS_ATTRS = {
    ATTR_BRIGHTNESS,
    ATTR_BRIGHTNESS_PCT,
    ATTR_BRIGHTNESS_STEP,
    ATTR_BRIGHTNESS_STEP_PCT,
}


def _get_switches_with_lights(
    hass: HomeAssistant, lights: list[str]
) -> list[AdaptiveSwitch]:
    """Get all switches that control at least one of the lights passed."""
    config_entries = hass.config_entries.async_entries(DOMAIN)
    data = hass.data[DOMAIN]
    switches = []
    for config in config_entries:
        entry = data.get(config.entry_id)
        if entry is None:  # entry might be disabled and therefore missing
            continue
        switch = data[config.entry_id]["instance"]
        all_check_lights = _expand_light_groups(hass, lights)
        switch._expand_light_groups()
        # Check if any of the lights are in the switch's lights
        if set(switch._lights) & set(all_check_lights):
            switches.append(switch)
    return switches


def find_switch_for_lights(
    hass: HomeAssistant,
    lights: list[str],
    is_on: bool = False,
) -> AdaptiveSwitch:
    """Find the switch that controls the lights in 'lights'."""
    switches = _get_switches_with_lights(hass, lights)
    if len(switches) == 1:
        return switches[0]
    elif len(switches) > 1:
        on_switches = [s for s in switches if s.is_on]
        if len(on_switches) == 1:
            # Of the multiple switches, only one is on
            return on_switches[0]
        raise ValueError(
            f"find_switch_for_lights: Light(s) {lights} found in multiple switch configs"
            f" ({[s.entity_id for s in switches]}). You must pass a switch under"
            f" 'entity_id'."
        )
    else:
        raise ValueError(
            f"find_switch_for_lights: Light(s) {lights} not found in any switch's"
            f" configuration. You must either include the light(s) that is/are"
            f" in the integration config, or pass a switch under 'entity_id'."
        )


# For documentation on this function, see integration_entities() from HomeAssistant Core:
# https://github.com/home-assistant/core/blob/dev/homeassistant/helpers/template.py#L1109
def _get_switches_from_service_call(
    hass: HomeAssistant, service_call: ServiceCall
) -> list[AdaptiveSwitch]:
    data = service_call.data
    lights = data[CONF_LIGHTS]
    switch_entity_ids: list[str] | None = data.get("entity_id")

    if not lights and not switch_entity_ids:
        raise ValueError(
            "adaptive-lighting: Neither a switch nor a light was provided in the service call."
            " If you intend to adapt all lights on all switches, please inform the developers at"
            " https://github.com/basnijholt/adaptive-lighting about your use case."
            " Currently, you must pass either an adaptive-lighting switch or the lights to an"
            " `adaptive_lighting` service call."
        )

    if switch_entity_ids is not None:
        if len(switch_entity_ids) > 1 and lights:
            raise ValueError(
                f"adaptive-lighting: Cannot pass multiple switches with lights argument."
                f" Invalid service data received: {service_call.data}"
            )
        switches = []
        ent_reg = entity_registry.async_get(hass)
        for entity_id in switch_entity_ids:
            ent_entry = ent_reg.async_get(entity_id)
            config_id = ent_entry.config_entry_id
            switches.append(hass.data[DOMAIN][config_id]["instance"])
        return switches

    if lights:
        switch = find_switch_for_lights(hass, lights, service_call)
        return [switch]

    raise ValueError(
        f"adaptive-lighting: Incorrect data provided in service call."
        f" Entities not found in the integration. Service data: {service_call.data}"
    )


async def handle_change_switch_settings(
    switch: AdaptiveSwitch, service_call: ServiceCall
) -> None:
    """Allows HASS to change config values via a service call."""
    data = service_call.data

    which = data.get(CONF_USE_DEFAULTS, "current")
    if which == "current":  # use whatever we're already using.
        defaults = switch._current_settings  # pylint: disable=protected-access
    elif which == "factory":  # use actual defaults listed in the documentation
        defaults = {key: default for key, default, _ in VALIDATION_TUPLES}
    elif which == "configuration":
        # use whatever's in the config flow or configuration.yaml
        defaults = switch._config_backup  # pylint: disable=protected-access
    else:
        defaults = None

    switch._set_changeable_settings(
        data=data,
        defaults=defaults,
    )

    _LOGGER.debug(
        "Called 'adaptive_lighting.change_switch_settings' service with '%s'",
        data,
    )

    all_lights = switch._lights  # pylint: disable=protected-access
    switch.turn_on_off_listener.reset(*all_lights, reset_manual_control=False)
    if switch.is_on:
        await switch._update_attrs_and_maybe_adapt_lights(  # pylint: disable=protected-access
            all_lights,
            transition=switch._initial_transition,
            force=True,
            context=switch.create_context("service", parent=service_call.context),
        )


def _fire_manual_control_event(
    switch: AdaptiveSwitch, light: Light, context: Context, is_async=True
):
    """Fire an event that 'light' is marked as manual_control."""
    hass = switch.hass
    fire = hass.bus.async_fire if is_async else hass.bus.fire
    _LOGGER.debug(
        "'adaptive_lighting.manual_control' event fired for %s for light %s",
        switch.entity_id,
        light.entity_id,
    )
    switch.turn_on_off_listener.mark_as_manual_control(light)
    fire(
        f"{DOMAIN}.manual_control",
        {ATTR_ENTITY_ID: light.entity_id, SWITCH_DOMAIN: switch.entity_id},
        context=context,
    )


async def async_setup_entry(
    hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: bool
):
    """Set up the AdaptiveLighting switch."""
    data = hass.data[DOMAIN]
    assert config_entry.entry_id in data

    if ATTR_TURN_ON_OFF_LISTENER not in data:
        data[ATTR_TURN_ON_OFF_LISTENER] = TurnOnOffListener(hass)
    turn_on_off_listener = data[ATTR_TURN_ON_OFF_LISTENER]
    sleep_mode_switch = SimpleSwitch(
        "Sleep Mode", False, hass, config_entry, ICON_SLEEP
    )
    adapt_color_switch = SimpleSwitch(
        "Adapt Color", True, hass, config_entry, ICON_COLOR_TEMP
    )
    adapt_brightness_switch = SimpleSwitch(
        "Adapt Brightness", True, hass, config_entry, ICON_BRIGHTNESS
    )
    switch = AdaptiveSwitch(
        hass,
        config_entry,
        turn_on_off_listener,
        sleep_mode_switch,
        adapt_color_switch,
        adapt_brightness_switch,
    )

    # save our switch instance, allows us to make switch's entity_id optional in service calls.
    hass.data[DOMAIN][config_entry.entry_id]["instance"] = switch

    data[config_entry.entry_id][SLEEP_MODE_SWITCH] = sleep_mode_switch
    data[config_entry.entry_id][ADAPT_COLOR_SWITCH] = adapt_color_switch
    data[config_entry.entry_id][ADAPT_BRIGHTNESS_SWITCH] = adapt_brightness_switch
    data[config_entry.entry_id][SWITCH_DOMAIN] = switch

    async_add_entities(
        [switch, sleep_mode_switch, adapt_color_switch, adapt_brightness_switch],
        update_before_add=True,
    )

    @callback
    async def handle_apply(service_call: ServiceCall):
        """Handle the entity service apply."""
        data = service_call.data
        _LOGGER.debug(
            "Called 'adaptive_lighting.apply' service with '%s'",
            data,
        )
        switches = _get_switches_from_service_call(hass, service_call)
        lights = data[CONF_LIGHTS]
        for switch in switches:
            if not lights:
                all_lights = switch._lights  # pylint: disable=protected-access
            else:
                all_lights = _expand_light_groups(switch.hass, lights)
            switch.turn_on_off_listener.lights.update(all_lights)
            for light in all_lights:
                if data[CONF_TURN_ON_LIGHTS] or is_on(hass, light):
                    await switch._adapt_light(  # pylint: disable=protected-access
                        light,
                        data[CONF_TRANSITION],
                        data[ATTR_ADAPT_BRIGHTNESS],
                        data[ATTR_ADAPT_COLOR],
                        data[CONF_PREFER_RGB_COLOR],
                        force=True,
                        context=switch.create_context(
                            "service", parent=service_call.context
                        ),
                    )

    @callback
    async def handle_set_manual_control(service_call: ServiceCall):
        """Set or unset lights as 'manually controlled'."""
        data = service_call.data
        _LOGGER.debug(
            "Called 'adaptive_lighting.set_manual_control' service with '%s'",
            data,
        )
        switches = _get_switches_from_service_call(hass, service_call)
        lights = data[CONF_LIGHTS]
        for switch in switches:
            if not lights:
                all_lights = switch._lights  # pylint: disable=protected-access
            else:
                all_lights = _expand_light_groups(switch.hass, lights)
            if service_call.data[CONF_MANUAL_CONTROL]:
                for light in all_lights:
                    _fire_manual_control_event(switch, light, service_call.context)
            else:
                switch.turn_on_off_listener.reset(*all_lights)
                if switch.is_on:
                    # pylint: disable=protected-access
                    await switch._update_attrs_and_maybe_adapt_lights(
                        all_lights,
                        transition=switch._initial_transition,
                        force=True,
                        context=switch.create_context(
                            "service", parent=service_call.context
                        ),
                    )

    # Register `apply` service
    hass.services.async_register(
        domain=DOMAIN,
        service=SERVICE_APPLY,
        service_func=handle_apply,
        schema=apply_service_schema(
            switch._initial_transition
        ),  # pylint: disable=protected-access
    )

    # Register `set_manual_control` service
    hass.services.async_register(
        domain=DOMAIN,
        service=SERVICE_SET_MANUAL_CONTROL,
        service_func=handle_set_manual_control,
        schema=SET_MANUAL_CONTROL_SCHEMA,
    )

    args = {vol.Optional(CONF_USE_DEFAULTS, default="current"): cv.string}
    # Modifying these after init isn't possible
    skip = (CONF_INTERVAL, CONF_NAME, CONF_LIGHTS)
    for k, _, valid in VALIDATION_TUPLES:
        if k not in skip:
            args[vol.Optional(k)] = valid
    platform = entity_platform.current_platform.get()
    platform.async_register_entity_service(
        SERVICE_CHANGE_SWITCH_SETTINGS,
        args,
        handle_change_switch_settings,
    )


def validate(
    config_entry: ConfigEntry,
    service_data: dict[str, Any] | None = None,
    defaults: dict[str, Any] | None = None,
):
    """Get the options and data from the config_entry and add defaults."""
    if defaults is None:
        data = {key: default for key, default, _ in VALIDATION_TUPLES}
    else:
        data = defaults

    if config_entry is not None:
        assert service_data is None
        assert defaults is None
        data.update(config_entry.options)  # come from options flow
        data.update(config_entry.data)  # all yaml settings come from data
    else:
        assert service_data is not None
        data.update(service_data)
    data = {key: replace_none_str(value) for key, value in data.items()}
    for key, (validate_value, _) in EXTRA_VALIDATION.items():
        value = data.get(key)
        if value is not None:
            data[key] = validate_value(value)  # Fix the types of the inputs
    return data


class AdaptiveSwitch(SwitchEntity, RestoreEntity):
    """Representation of a Adaptive Lighting switch."""

    def __init__(
        self,
        hass,
        config_entry: ConfigEntry,
        turn_on_off_listener: TurnOnOffListener,
        sleep_mode_switch: SimpleSwitch,
        adapt_color_switch: SimpleSwitch,
        adapt_brightness_switch: SimpleSwitch,
    ):
        """Initialize the Adaptive Lighting switch."""
        # Set attributes that can't be modified during runtime
        self.hass = hass
        self.turn_on_off_listener = turn_on_off_listener
        self.sleep_mode_switch = sleep_mode_switch
        self.adapt_color_switch = adapt_color_switch
        self.adapt_brightness_switch = adapt_brightness_switch

        data = validate(config_entry)

        self._name = data[CONF_NAME]
        self._interval = data[CONF_INTERVAL]
        self._lights = data[CONF_LIGHTS]

        # backup data for use in change_switch_settings "configuration" CONF_USE_DEFAULTS
        self._config_backup = deepcopy(data)
        self._set_changeable_settings(
            data=data,
            defaults=None,
        )

        # Set other attributes
        self._icon = ICON_MAIN
        self._state = None

        # Tracks 'off' → 'on' state changes
        self._on_to_off_event: dict[str, Event] = {}
        # Tracks 'on' → 'off' state changes
        self._off_to_on_event: dict[str, Event] = {}
        # Locks that prevent light adjusting when waiting for a light to 'turn_off'
        self._locks: dict[str, asyncio.Lock] = {}
        # To count the number of `Context` instances
        self._context_cnt: int = 0

        # Set in self._update_attrs_and_maybe_adapt_lights
        self._settings: dict[str, Any] = {}

        # Set and unset tracker in async_turn_on and async_turn_off
        self.remove_listeners = []
        _LOGGER.debug(
            "%s: Setting up with '%s',"
            " config_entry.data: '%s',"
            " config_entry.options: '%s', converted to '%s'.",
            self._name,
            self._lights,
            config_entry.data,
            config_entry.options,
            data,
        )

    def _set_changeable_settings(
        self,
        data: dict,
        defaults: dict,
    ):
        # Only pass settings users can change during runtime
        data = validate(
            config_entry=None,
            service_data=data,
            defaults=defaults,
        )

        # backup data for use in change_switch_settings "current" CONF_USE_DEFAULTS
        self._current_settings = data

        self._detect_non_ha_changes = data[CONF_DETECT_NON_HA_CHANGES]
        self._include_config_in_attributes = data[CONF_INCLUDE_CONFIG_IN_ATTRIBUTES]
        self._config: dict[str, Any] = {}
        if self._include_config_in_attributes:
            attrdata = deepcopy(data)
            for k, v in attrdata.items():
                if isinstance(v, (datetime.date, datetime.datetime)):
                    attrdata[k] = v.isoformat()
                if isinstance(v, (datetime.timedelta)):
                    attrdata[k] = v.total_seconds()
            self._config.update(attrdata)

        self._initial_transition = data[CONF_INITIAL_TRANSITION]
        self._sleep_transition = data[CONF_SLEEP_TRANSITION]
        self._only_once = data[CONF_ONLY_ONCE]
        self._prefer_rgb_color = data[CONF_PREFER_RGB_COLOR]
        self._separate_turn_on_commands = data[CONF_SEPARATE_TURN_ON_COMMANDS]
        self._transition = data[CONF_TRANSITION]
        self._adapt_delay = data[CONF_ADAPT_DELAY]
        self._send_split_delay = data[CONF_SEND_SPLIT_DELAY]
        self._take_over_control = data[CONF_TAKE_OVER_CONTROL]
        self._detect_non_ha_changes = data[CONF_DETECT_NON_HA_CHANGES]
        if not data[CONF_TAKE_OVER_CONTROL] and data[CONF_DETECT_NON_HA_CHANGES]:
            _LOGGER.warning(
                "%s: Config mismatch: 'detect_non_ha_changes: true' "
                "requires 'take_over_control' to be enabled. Adjusting config "
                "and continuing setup with `take_over_control: true`.",
                self._name,
            )
            self._take_over_control = True
        self._auto_reset_manual_control_time = data[CONF_AUTORESET_CONTROL]
        self._expand_light_groups()  # updates manual control timers
        _loc = get_astral_location(self.hass)
        if isinstance(_loc, tuple):
            # Astral v2.2
            location, _ = _loc
        else:
            # Astral v1
            location = _loc

        self._sun_light_settings = SunLightSettings(
            name=self._name,
            astral_location=location,
            adapt_until_sleep=data[CONF_ADAPT_UNTIL_SLEEP],
            max_brightness=data[CONF_MAX_BRIGHTNESS],
            max_color_temp=data[CONF_MAX_COLOR_TEMP],
            min_brightness=data[CONF_MIN_BRIGHTNESS],
            min_color_temp=data[CONF_MIN_COLOR_TEMP],
            sleep_brightness=data[CONF_SLEEP_BRIGHTNESS],
            sleep_color_temp=data[CONF_SLEEP_COLOR_TEMP],
            sleep_rgb_color=data[CONF_SLEEP_RGB_COLOR],
            sleep_rgb_or_color_temp=data[CONF_SLEEP_RGB_OR_COLOR_TEMP],
            sunrise_offset=data[CONF_SUNRISE_OFFSET],
            sunrise_time=data[CONF_SUNRISE_TIME],
            max_sunrise_time=data[CONF_MAX_SUNRISE_TIME],
            sunset_offset=data[CONF_SUNSET_OFFSET],
            sunset_time=data[CONF_SUNSET_TIME],
            min_sunset_time=data[CONF_MIN_SUNSET_TIME],
            time_zone=self.hass.config.time_zone,
            transition=data[CONF_TRANSITION],
        )
        _LOGGER.debug(
            "%s: Set switch settings for lights '%s'. now using data: '%s'",
            self._name,
            self._lights,
            data,
        )

    @property
    def name(self):
        """Return the name of the device if any."""
        return f"Adaptive Lighting: {self._name}"

    @property
    def unique_id(self):
        """Return the unique ID of entity."""
        return self._name

    @property
    def is_on(self) -> bool | None:
        """Return true if adaptive lighting is on."""
        return self._state

    async def async_added_to_hass(self) -> None:
        """Call when entity about to be added to hass."""
        if self.hass.is_running:
            await self._setup_listeners()
        else:
            self.hass.bus.async_listen_once(
                EVENT_HOMEASSISTANT_STARTED, self._setup_listeners
            )
        last_state = await self.async_get_last_state()
        is_new_entry = last_state is None  # newly added to HA
        if is_new_entry or last_state.state == STATE_ON:
            await self.async_turn_on(adapt_lights=not self._only_once)
        else:
            self._state = False
            assert not self.remove_listeners

    async def async_will_remove_from_hass(self):
        """Remove the listeners upon removing the component."""
        self._remove_listeners()

    def _expand_light_groups(self) -> None:
        all_lights = _expand_light_groups(self.hass, self._lights)
        self.turn_on_off_listener.lights.update(all_lights)
        self.turn_on_off_listener.set_auto_reset_manual_control_times(
            all_lights, self._auto_reset_manual_control_time
        )
        self._lights = list(all_lights)

    async def _setup_listeners(self, _=None) -> None:
        _LOGGER.debug("%s: Called '_setup_listeners'", self._name)
        if not self.is_on or not self.hass.is_running:
            _LOGGER.debug("%s: Cancelled '_setup_listeners'", self._name)
            return

        assert not self.remove_listeners

        remove_interval = async_track_time_interval(
            self.hass, self._async_update_at_interval, self._interval
        )
        remove_sleep = async_track_state_change_event(
            self.hass,
            self.sleep_mode_switch.entity_id,
            self._sleep_mode_switch_state_event,
        )

        self.remove_listeners.extend([remove_interval, remove_sleep])

        if self._lights:
            self._expand_light_groups()
            remove_state = async_track_state_change_event(
                self.hass, self._lights, self._light_event
            )
            self.remove_listeners.append(remove_state)

    def _remove_listeners(self) -> None:
        while self.remove_listeners:
            remove_listener = self.remove_listeners.pop()
            remove_listener()

    @property
    def icon(self) -> str:
        """Icon to use in the frontend, if any."""
        return self._icon

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the attributes of the switch."""
        extra_state_attributes = {"configuration": self._config}
        if not self.is_on:
            for key in self._settings:
                extra_state_attributes[key] = None
            return extra_state_attributes
        extra_state_attributes["manual_control"] = [
            light
            for light in self._lights
            if self.turn_on_off_listener.manual_control.get(light)
        ]
        extra_state_attributes.update(self._settings)
        timers = self.turn_on_off_listener.auto_reset_manual_control_timers
        extra_state_attributes["autoreset_time_remaining"] = {
            light: time
            for light in self._lights
            if (timer := timers.get(light)) and (time := timer.remaining_time()) > 0
        }
        return extra_state_attributes

    def create_context(
        self, which: str = "default", parent: Context | None = None
    ) -> Context:
        """Create a context that identifies this Adaptive Lighting instance."""
        # Right now the highest number of each context_id it can create is
        # 'adapt_lgt:XXXX:turn_on:*************'
        # 'adapt_lgt:XXXX:interval:************'
        # 'adapt_lgt:XXXX:adapt_lights:********'
        # 'adapt_lgt:XXXX:sleep:***************'
        # 'adapt_lgt:XXXX:light_event:*********'
        # 'adapt_lgt:XXXX:service:*************'
        # The smallest space we have is for adapt_lights, which has
        # 8 characters. In base85 encoding, that's enough space to hold values
        # up to 2**48 - 1, which should give us plenty of calls before we wrap.
        context = create_context(self._name, which, self._context_cnt, parent=parent)
        self._context_cnt += 1
        return context

    async def async_turn_on(  # pylint: disable=arguments-differ
        self, adapt_lights: bool = True
    ) -> None:
        """Turn on adaptive lighting."""
        _LOGGER.debug(
            "%s: Called 'async_turn_on', current state is '%s'", self._name, self._state
        )
        if self.is_on:
            return
        self._state = True
        self.turn_on_off_listener.reset(*self._lights)
        await self._setup_listeners()
        if adapt_lights:
            await self._update_attrs_and_maybe_adapt_lights(
                transition=self._initial_transition,
                force=True,
                context=self.create_context("turn_on"),
            )

    async def async_turn_off(self, **kwargs) -> None:
        """Turn off adaptive lighting."""
        if not self.is_on:
            return
        self._state = False
        self._remove_listeners()
        self.turn_on_off_listener.reset(*self._lights)

    async def _async_update_at_interval(self, now=None) -> None:
        await self._update_attrs_and_maybe_adapt_lights(
            transition=self._transition,
            force=False,
            context=self.create_context("interval"),
        )

    async def _adapt_light(
        self,
        light_id: str,
        transition: int | None = None,
        adapt_brightness: bool | None = None,
        adapt_color: bool | None = None,
        prefer_rgb_color: bool | None = None,
        force: bool = False,
        context: Context | None = None,
    ) -> None:
        lock = self._locks.get(light_id)
        if lock is not None and lock.locked():
            _LOGGER.debug("%s: '%s' is locked", self._name, light_id)
            return
        if transition is None:
            transition = self._transition
        if adapt_brightness is None:
            adapt_brightness = self.adapt_brightness_switch.is_on
        if adapt_color is None:
            adapt_color = self.adapt_color_switch.is_on
        if prefer_rgb_color is None:
            prefer_rgb_color = self._prefer_rgb_color

        # The switch might be off and not have _settings set.
        self._settings = self._sun_light_settings.calculate(
            self.sleep_mode_switch.is_on, transition
        )

        # Build service data.
        service_data = {ATTR_ENTITY_ID: light_id}
        features = _supported_features(self.hass, light_id)

        # Check transition == 0 to fix #378
        if "transition" in features and transition > 0:
            service_data[ATTR_TRANSITION] = transition
        if "brightness" in features and adapt_brightness:
            brightness = round(255 * self._settings["brightness_pct"] / 100)
            service_data[ATTR_BRIGHTNESS] = brightness

        sleep_rgb = (
            self.sleep_mode_switch.is_on
            and self._sun_light_settings.sleep_rgb_or_color_temp == "rgb_color"
        )
        if (
            "color_temp" in features
            and adapt_color
            and not (prefer_rgb_color and "color" in features)
            and not (sleep_rgb and "color" in features)
        ):
            _LOGGER.debug("%s: Setting color_temp of light %s", self._name, light_id)
            attributes = self.hass.states.get(light_id).attributes
            min_kelvin = attributes["min_color_temp_kelvin"]
            max_kelvin = attributes["max_color_temp_kelvin"]
            color_temp_kelvin = self._settings["color_temp_kelvin"]
            color_temp_kelvin = max(min(color_temp_kelvin, max_kelvin), min_kelvin)
            service_data[ATTR_COLOR_TEMP_KELVIN] = color_temp_kelvin
        elif "color" in features and adapt_color:
            _LOGGER.debug("%s: Setting rgb_color of light %s", self._name, light_id)
            service_data[ATTR_RGB_COLOR] = self._settings["rgb_color"]

        context = context or self.create_context("adapt_lights")

        # See #80. Doesn't check if transitions differ but it does the job.
        light = self.turn_on_off_listener.lights.get(light_id)
        if not force and light.last_service_data == service_data:
            _LOGGER.debug(
                "%s: Cancelling adapt to light %s, there's no new values to set (context.id='%s')",
                self._name,
                light_id,
                context.id,
            )
            return
        else:
            light.last_service_data = service_data

        async def turn_on(service_data):
            _LOGGER.debug(
                "%s: Scheduling 'light.turn_on' with the following 'service_data': %s"
                " with context.id='%s'",
                self._name,
                service_data,
                context.id,
            )
            await self.hass.services.async_call(
                LIGHT_DOMAIN,
                SERVICE_TURN_ON,
                service_data,
                context=context,
            )

        async def turn_on_split():
            # Could be a list of length 1 or 2
            service_datas = _split_service_data(
                service_data, adapt_brightness, adapt_color
            )
            await turn_on(service_datas[0])
            if len(service_datas) == 2:
                transition = service_datas[0].get(ATTR_TRANSITION)
                if transition is not None:
                    await asyncio.sleep(transition)
                await asyncio.sleep(self._send_split_delay / 1000.0)
                await turn_on(service_datas[1])

        if not self._separate_turn_on_commands:
            await turn_on(service_data)
        else:
            if (
                previous_task := self.turn_on_off_listener.split_adaptation_tasks.get(
                    light_id
                )
            ) is not None:
                previous_task.cancel()

            try:
                task = self.turn_on_off_listener.split_adaptation_tasks[
                    light_id
                ] = asyncio.ensure_future(turn_on_split())
                await task
            except asyncio.CancelledError:
                _LOGGER.debug("Split adaptation of %s cancelled", light_id)

    async def _update_attrs_and_maybe_adapt_lights(
        self,
        lights: list[str] | None = None,
        transition: int | None = None,
        force: bool = False,
        context: Context | None = None,
    ) -> None:
        assert context is not None
        _LOGGER.debug(
            "%s: '_update_attrs_and_maybe_adapt_lights' called with context.id='%s'",
            self._name,
            context.id,
        )
        assert self.is_on
        self._settings.update(
            self._sun_light_settings.calculate(self.sleep_mode_switch.is_on, transition)
        )
        self.async_write_ha_state()

        if lights is None:
            lights = self._lights

        filtered_lights = []
        if not force:
            if self._only_once:
                return
            for light in lights:
                # Don't adapt lights that haven't finished prior transitions.
                timer = self.turn_on_off_listener.transition_timers.get(light)
                if timer is not None and timer.is_running():
                    _LOGGER.debug(
                        "%s: Light '%s' is still transitioning",
                        self._name,
                        light,
                    )
                else:
                    filtered_lights.append(light)
        else:
            filtered_lights = lights

        if not filtered_lights:
            return

        await self._update_manual_control_and_maybe_adapt(
            filtered_lights, transition, force, context
        )

    async def _update_manual_control_and_maybe_adapt(
        self,
        lights: list[str],
        transition: int | None,
        force: bool,
        context: Context | None,
    ) -> None:
        assert context is not None
        _LOGGER.debug(
            "%s: '_update_manual_control_and_maybe_adapt(%s, %s, force=%s, context.id=%s)' called",
            self.name,
            lights,
            transition,
            force,
            context.id,
        )

        adapt_brightness = self.adapt_brightness_switch.is_on
        adapt_color = self.adapt_color_switch.is_on

        for light in lights:
            if not is_on(self.hass, light):
                continue

            manually_controlled = self.turn_on_off_listener.is_manually_controlled(
                self,
                light,
                force,
                adapt_brightness,
                adapt_color,
            )

            significant_change = (
                self._detect_non_ha_changes
                and not force
                and await self.turn_on_off_listener.significant_change(
                    self,
                    light,
                    adapt_brightness,
                    adapt_color,
                    context,
                )
            )

            if self._take_over_control and (manually_controlled or significant_change):
                if manually_controlled:
                    _LOGGER.debug(
                        "%s: '%s' is being manually controlled, stop adapting, context.id=%s.",
                        self._name,
                        light,
                        context.id,
                    )
                else:
                    _fire_manual_control_event(self, light, context)
            else:
                await self._adapt_light(light, transition, force=force, context=context)

    async def _sleep_mode_switch_state_event(self, event: Event) -> None:
        if not match_switch_state_event(event, (STATE_ON, STATE_OFF)):
            _LOGGER.debug("%s: Ignoring sleep event %s", self._name, event)
            return
        _LOGGER.debug(
            "%s: _sleep_mode_switch_state_event, event: '%s'", self._name, event
        )
        # Reset the manually controlled status when the "sleep mode" changes
        self.turn_on_off_listener.reset(*self._lights)
        await self._update_attrs_and_maybe_adapt_lights(
            transition=self._sleep_transition,
            force=True,
            context=self.create_context("sleep", parent=event.context),
        )

    async def _light_event(self, event: Event) -> None:
        old_state = event.data.get("old_state")
        new_state = event.data.get("new_state")
        entity_id = event.data.get("entity_id")
        if (
            old_state is not None
            and old_state.state == STATE_OFF
            and new_state is not None
            and new_state.state == STATE_ON
        ):
            _LOGGER.debug(
                "%s: Detected an 'off' → 'on' event for '%s' with context.id='%s'",
                self._name,
                entity_id,
                event.context.id,
            )
            self.turn_on_off_listener.reset(entity_id, reset_manual_control=False)
            # Tracks 'off' → 'on' state changes
            self._off_to_on_event[entity_id] = event
            lock = self._locks.get(entity_id)
            if lock is None:
                lock = self._locks[entity_id] = asyncio.Lock()
            async with lock:
                if await self.turn_on_off_listener.maybe_cancel_adjusting(
                    entity_id,
                    off_to_on_event=event,
                    on_to_off_event=self._on_to_off_event.get(entity_id),
                ):
                    # Stop if a rapid 'off' → 'on' → 'off' happens.
                    _LOGGER.debug(
                        "%s: Cancelling adjusting lights for %s", self._name, entity_id
                    )
                    return

            if self._adapt_delay > 0:
                _LOGGER.debug(
                    "%s: sleep started for '%s' with context.id='%s'",
                    self._name,
                    entity_id,
                    event.context.id,
                )
                await asyncio.sleep(self._adapt_delay)
                _LOGGER.debug(
                    "%s: sleep ended for '%s' with context.id='%s'",
                    self._name,
                    entity_id,
                    event.context.id,
                )

            await self._update_attrs_and_maybe_adapt_lights(
                lights=[entity_id],
                transition=self._initial_transition,
                force=True,
                context=self.create_context("light_event", parent=event.context),
            )
        elif (
            old_state is not None
            and old_state.state == STATE_ON
            and new_state is not None
            and new_state.state == STATE_OFF
        ):
            # Tracks 'off' → 'on' state changes
            self._on_to_off_event[entity_id] = event
            self.turn_on_off_listener.reset(entity_id)


class SimpleSwitch(SwitchEntity, RestoreEntity):
    """Representation of a Adaptive Lighting switch."""

    def __init__(
        self,
        which: str,
        initial_state: bool,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        icon: str,
    ):
        """Initialize the Adaptive Lighting switch."""
        self.hass = hass
        data = validate(config_entry)
        self._icon = icon
        self._state = None
        self._which = which
        name = data[CONF_NAME]
        self._unique_id = f"{name}_{slugify(self._which)}"
        self._name = f"Adaptive Lighting {which}: {name}"
        self._initial_state = initial_state

    @property
    def name(self):
        """Return the name of the device if any."""
        return self._name

    @property
    def unique_id(self):
        """Return the unique ID of entity."""
        return self._unique_id

    @property
    def icon(self) -> str:
        """Icon to use in the frontend, if any."""
        return self._icon

    @property
    def is_on(self) -> bool | None:
        """Return true if adaptive lighting is on."""
        return self._state

    async def async_added_to_hass(self) -> None:
        """Call when entity about to be added to hass."""
        last_state = await self.async_get_last_state()
        _LOGGER.debug("%s: last state is %s", self._name, last_state)
        if (last_state is None and self._initial_state) or (
            last_state is not None and last_state.state == STATE_ON
        ):
            await self.async_turn_on()
        else:
            await self.async_turn_off()

    async def async_turn_on(self, **kwargs) -> None:
        """Turn on adaptive lighting sleep mode."""
        _LOGGER.debug("%s: Turning on", self._name)
        self._state = True

    async def async_turn_off(self, **kwargs) -> None:
        """Turn off adaptive lighting sleep mode."""
        _LOGGER.debug("%s: Turning off", self._name)
        self._state = False


class TurnOnOffListener:
    """Track 'light.turn_off' and 'light.turn_on' service calls."""

    def __init__(self, hass: HomeAssistant):
        """Initialize the TurnOnOffListener that is shared among all switches."""
        self.hass = hass
        self.lights: dict[str, Light] = {}

        self.remove_listener = self.hass.bus.async_listen(
            EVENT_CALL_SERVICE, self.turn_on_off_event_listener
        )
        self.remove_listener2 = self.hass.bus.async_listen(
            EVENT_STATE_CHANGED, self.state_changed_event_listener
        )

    def _handle_timer(
        self,
        light: Light,
        timer_name: str,
        delay: float | None,
        reset_coroutine: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        timer = light[timer_name]
        if timer is not None:
            if delay is None:  # Timer object exists, but should not anymore
                timer.cancel()
                light[timer_name] = None
            else:  # Timer object already exists, just update the delay and restart it
                timer.delay = delay
                timer.start()
        elif delay is not None:  # Timer object does not exist, create it
            timer = _AsyncSingleShotTimer(delay, reset_coroutine)
            light[timer_name] = timer
            timer.start()

    def start_transition_timer(self, light: Light) -> None:
        """Mark a light as manually controlled."""
        last_service_data = light.last_service_data
        if not last_service_data:
            _LOGGER.debug("This should not ever happen. Please report to the devs.")
            return
        last_transition = last_service_data.get(ATTR_TRANSITION)
        if not last_transition:
            _LOGGER.debug(
                "No transition in last adapt for light %s, continuing...", light
            )
            return
        _LOGGER.debug(
            "Start transition timer of %s seconds for light %s", last_transition, light
        )

        async def reset():
            ValueError("TEST")
            _LOGGER.debug(
                "Transition finished for light %s",
                light,
            )

        self._handle_timer(light, "transition_timer", last_transition, reset)

    def set_auto_reset_manual_control_times(self, lights: list[Light], time: float):
        """Set the time after which the lights are automatically reset."""
        if time == 0:
            return
        for light in lights:
            old_time = light.auto_reset_manual_control_time
            if (old_time is not None) and (old_time != time):
                _LOGGER.info(
                    "Setting auto_reset_manual_control for '%s' from %s seconds to %s seconds."
                    " This might happen because the light is in multiple swiches"
                    " or because of a config change.",
                    light,
                    old_time,
                    time,
                )
            light.auto_reset_manual_control_time = time

    def mark_as_manual_control(self, light: Light) -> None:
        """Mark a light as manually controlled."""
        _LOGGER.debug("Marking '%s' as manually controlled.", light.entity_id)
        light.manual_control = True
        delay = light.auto_reset_manual_control_time

        async def reset():
            self.reset(light)
            switches = _get_switches_with_lights(self.hass, [light])
            for switch in switches:
                if not switch.is_on:
                    continue
                await switch._update_attrs_and_maybe_adapt_lights(
                    [light],
                    transition=switch._initial_transition,
                    force=True,
                    context=switch.create_context("autoreset"),
                )
            _LOGGER.debug(
                "Auto resetting 'manual_control' status of '%s' because"
                " it was not manually controlled for %s seconds.",
                light,
                delay,
            )
            assert not light.manual_control

        self._handle_timer(light, "auto_reset_manual_control_timer", delay, reset)

    def reset(self, *lights: Light, reset_manual_control=True) -> None:
        """Reset the 'manual_control' status of the lights."""
        for light in lights:
            light.reset(reset_manual_control)

    async def turn_on_off_event_listener(self, event: Event) -> None:
        """Track 'light.turn_off' and 'light.turn_on' service calls."""
        if not is_light_on_off_event(event):
            return

        entity_ids = get_entity_ids_from_service_event(self.hass, event)

        if not entity_ids or not any(eid in self.lights for eid in entity_ids):
            return

        service = event.data[ATTR_SERVICE]
        service_data = event.data[ATTR_SERVICE_DATA]

        if service == SERVICE_TURN_OFF:
            transition = service_data.get(ATTR_TRANSITION)
            _LOGGER.debug(
                "Detected an 'light.turn_off('%s', transition=%s)' event with context.id='%s'",
                entity_ids,
                transition,
                event.context.id,
            )
            for eid in entity_ids:
                self.lights[eid].process_turn_off_event(event)

        elif service == SERVICE_TURN_ON:
            _LOGGER.debug(
                "Detected an 'light.turn_on('%s')' event with context.id='%s'",
                entity_ids,
                event.context.id,
            )
            for eid in entity_ids:
                self.lights[eid].process_turn_on_event(event)

    async def state_changed_event_listener(self, event: Event) -> None:
        """Track 'state_changed' events."""
        entity_id = event.data.get(ATTR_ENTITY_ID, "")
        if entity_id not in self.lights:
            return
        light = self.lights[entity_id]

        new_state = event.data.get("new_state")
        if new_state is not None and new_state.state == STATE_ON:
            _LOGGER.debug(
                "Detected a '%s' 'state_changed' event: '%s' with context.id='%s'",
                entity_id,
                new_state.attributes,
                new_state.context.id,
            )

        if new_state is not None and new_state.state == STATE_ON:
            # It is possible to have multiple state change events with the same context.
            # This can happen because a `turn_on.light(brightness_pct=100, transition=30)`
            # event leads to an instant state change of
            # `new_state=dict(brightness=100, ...)`. However, after polling the light
            # could still only be `new_state=dict(brightness=50, ...)`.
            # We save all events because the first event change might indicate at what
            # settings the light will be later *or* the second event might indicate a
            # final state. The latter case happens for example when a light was
            # called with a color_temp outside of its range (and HA reports the
            # incorrect 'min_kelvin' and 'max_kelvin', which happens e.g., for
            # Philips Hue White GU10 Bluetooth lights).
            old_state: list[State] | None = light.last_state_change
            if is_our_context(new_state.context):
                if (
                    old_state is not None
                    and old_state[0].context.id == new_state.context.id
                ):
                    _LOGGER.debug(
                        "TurnOnOffListener: State change event of '%s' is already"
                        " in 'self.last_state_change' (%s)"
                        " adding this state also",
                        entity_id,
                        new_state.context.id,
                    )
                    light.last_state_change.append(new_state)
                else:
                    _LOGGER.debug(
                        "TurnOnOffListener: New adapt '%s' found for %s",
                        new_state,
                        entity_id,
                    )
                    light.last_state_change = [new_state]
                    _LOGGER.debug(
                        "Last transition: %s",
                        light.last_service_data.get(ATTR_TRANSITION),
                    )
                    self.start_transition_timer(light)
            elif old_state is not None:
                light.last_state_change.append(new_state)

    def is_manually_controlled(
        self,
        switch: AdaptiveSwitch,
        light: Light,
        force: bool,
        adapt_brightness: bool,
        adapt_color: bool,
    ) -> bool:
        """Check if the light has been 'on' and is now manually controlled."""
        if light.manual_control:
            # Manually controlled until light is turned on and off
            return True

        turn_on_event = light.turn_on_event
        if (
            turn_on_event is not None
            and not is_our_context(turn_on_event.context)
            and not force
        ):
            keys = turn_on_event.data[ATTR_SERVICE_DATA].keys()
            if (adapt_color and COLOR_ATTRS.intersection(keys)) or (
                adapt_brightness and BRIGHTNESS_ATTRS.intersection(keys)
            ):
                # Light was already on and 'light.turn_on' was not called by
                # the adaptive_lighting integration.
                light.manual_control = True
                _fire_manual_control_event(switch, light, turn_on_event.context)
                _LOGGER.debug(
                    "'%s' was already on and 'light.turn_on' was not called by the"
                    " adaptive_lighting integration (context.id='%s'), the Adaptive"
                    " Lighting will stop adapting the light until the switch or the"
                    " light turns off and then on again.",
                    light,
                    turn_on_event.context.id,
                )
        return light.manual_control

    async def significant_change(
        self,
        switch: AdaptiveSwitch,
        light: Light,
        adapt_brightness: bool,
        adapt_color: bool,
        context: Context,
    ) -> bool:
        """Has the light made a significant change since last update.

        This method will detect changes that were made to the light without
        calling 'light.turn_on', so outside of Home Assistant. If a change is
        detected, we mark the light as 'manually controlled' until the light
        or switch is turned 'off' and 'on' again.
        """
        if light.last_service_data is None:
            return

        compare_to = functools.partial(
            _attributes_have_changed,
            light=light,
            adapt_brightness=adapt_brightness,
            adapt_color=adapt_color,
            context=context,
        )
        # Update state and check for a manual change not done in HA.
        # Ensure HASS is correctly updating your light's state with
        # light.turn_on calls if any problems arise. This
        # can happen e.g. using zigbee2mqtt with 'report: false' in device settings.
        if switch._detect_non_ha_changes:
            _LOGGER.debug(
                "%s: 'detect_non_ha_changes: true', calling update_entity(%s)"
                " and check if it's last adapt succeeded.",
                switch._name,
                light,
            )
            # This update_entity probably isn't necessary now that we're checking
            # if transitions finished from our last adapt.
            await self.hass.helpers.entity_component.async_update_entity(light)
            refreshed_state = self.hass.states.get(light)
            _LOGGER.debug(
                "%s: Current state of %s: %s",
                switch._name,
                light,
                refreshed_state,
            )
            changed = compare_to(
                old_attributes=light.last_service_data,
                new_attributes=refreshed_state.attributes,
            )
            if changed:
                _LOGGER.debug(
                    "State of '%s' didn't change wrt 'last_service_data' (context.id=%s)",
                    light,
                    context.id,
                )
                return True
        _LOGGER.debug(
            "%s: Light '%s' correctly matches our last adapt's service data, continuing..."
            " context.id=%s.",
            switch._name,
            light,
            context.id,
        )
        return False

    async def maybe_cancel_adjusting(
        self, light: Light, off_to_on_event: Event, on_to_off_event: Event | None
    ) -> bool:
        """Cancel the adjusting of a light if it has just been turned off.

        Possibly the lights just got a 'turn_off' call, however, the light
        is actually still turning off (e.g., because of a 'transition') and
        HA polls the light before the light is 100% off. This might trigger
        a rapid switch 'off' → 'on' → 'off'. To prevent this component
        from interfering on the 'on' state, we make sure to wait at least
        TURNING_OFF_DELAY (or the 'turn_off' transition time) between a
        'off' → 'on' event and then check whether the light is still 'on' or
        if the brightness is still decreasing. Only if it is the case we
        adjust the lights.
        """
        if on_to_off_event is None:
            # No state change has been registered before.
            return False

        id_on_to_off = on_to_off_event.context.id

        turn_off_event = light.turn_off_event
        if turn_off_event is not None:
            transition = turn_off_event.data[ATTR_SERVICE_DATA].get(ATTR_TRANSITION)
        else:
            transition = None

        turn_on_event = light.turn_on_event
        if turn_on_event is None:
            # This means that the light never got a 'turn_on' call that we
            # registered. I am not 100% sure why this happens, but it does.
            # This is a fix for #170 and #232.
            return False
        id_turn_on = turn_on_event.context.id

        id_off_to_on = off_to_on_event.context.id

        if id_off_to_on == id_turn_on and id_off_to_on is not None:
            # State change 'off' → 'on' triggered by 'light.turn_on'.
            return False

        if (
            turn_off_event is not None
            and id_on_to_off == turn_off_event.context.id
            and id_on_to_off is not None
            and transition is not None  # 'turn_off' is called with transition=...
        ):
            # State change 'on' → 'off' and 'light.turn_off(..., transition=...)' come
            # from the same event, so wait at least the 'turn_off' transition time.
            delay = max(transition, TURNING_OFF_DELAY)
        else:
            # State change 'off' → 'on' happened because the light state was set.
            # Possibly because of polling.
            delay = TURNING_OFF_DELAY

        delta_time = (dt_util.utcnow() - on_to_off_event.time_fired).total_seconds()
        if delta_time > delay:
            return False

        # Here we could just `return True` but because we want to prevent any updates
        # from happening to this light (through async_track_time_interval or
        # sleep_state) for some time, we wait below until the light
        # is 'off' or the time has passed.

        delay -= delta_time  # delta_time has passed since the 'off' → 'on' event
        _LOGGER.debug("Waiting with adjusting '%s' for %s", light.entity_id, delay)

        for _ in range(3):
            # It can happen that the actual transition time is longer than the
            # specified time in the 'turn_off' service.
            coro = asyncio.sleep(delay)
            task = light.sleep_task = asyncio.ensure_future(coro)
            try:
                await task
            except asyncio.CancelledError:  # 'light.turn_on' has been called
                _LOGGER.debug(
                    "Sleep task is cancelled due to 'light.turn_on('%s')' call",
                    light.entity_id,
                )
                return False

            if not is_on(self.hass, light.entity_id):
                return True
            delay = TURNING_OFF_DELAY  # next time only wait this long

        if transition is not None:
            # Always ignore when there's a 'turn_off' transition.
            # Because it seems like HA cannot detect whether a light is
            # transitioning into 'off'. Maybe needs some discussion/input?
            return True

        # Now we assume that the lights are still on and they were intended
        # to be on. In case this still gives problems for some, we might
        # choose to **only** adapt on 'light.turn_on' events and ignore
        # other 'off' → 'on' state switches resulting from polling. That
        # would mean we 'return True' here.
        return False

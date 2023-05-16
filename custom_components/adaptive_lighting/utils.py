import base64
import ulid_transform
import asyncio
import math
import logging
from typing import Any

from homeassistant.core import (
    Context,
    Event,
    HomeAssistant,
)
from homeassistant.const import (
    ATTR_SUPPORTED_FEATURES,
    ATTR_DOMAIN,
    ATTR_SERVICE,
    ATTR_SERVICE_DATA,
    ATTR_ENTITY_ID,
    ATTR_AREA_ID,
    SERVICE_TURN_ON,
    SERVICE_TURN_OFF,
)
from homeassistant.components.light import (
    SUPPORT_BRIGHTNESS,
    SUPPORT_COLOR,
    SUPPORT_COLOR_TEMP,
    SUPPORT_TRANSITION,
    ATTR_BRIGHTNESS,
    ATTR_COLOR_TEMP_KELVIN,
    ATTR_RGB_COLOR,
    ATTR_TRANSITION,
    ATTR_SUPPORTED_COLOR_MODES,
    COLOR_MODE_RGB,
    COLOR_MODE_RGBW,
    COLOR_MODE_XY,
    COLOR_MODE_HS,
    COLOR_MODE_COLOR_TEMP,
    COLOR_MODE_BRIGHTNESS,
    ATTR_XY_COLOR,
    DOMAIN as LIGHT_DOMAIN,
)
from homeassistant.util.color import (
    color_temperature_to_rgb,
    color_xy_to_RGB,
)
import homeassistant.util.dt as dt_util
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.template import area_entities

from .const import (
    ATTR_TURN_ON_OFF_LISTENER,
    DOMAIN,
)

_SUPPORT_OPTS = {
    "brightness": SUPPORT_BRIGHTNESS,
    "color_temp": SUPPORT_COLOR_TEMP,
    "color": SUPPORT_COLOR,
    "transition": SUPPORT_TRANSITION,
}

# Consider it a significant change when attribute changes more than
BRIGHTNESS_CHANGE = 25  # ≈10% of total range
COLOR_TEMP_CHANGE = 100  # ≈3% of total range (2000-6500)
RGB_REDMEAN_CHANGE = 80  # ≈10% of total range

# Keep a short domain version for the context instances (which can only be 36 chars)
_DOMAIN_SHORT = "al"

_LOGGER = logging.getLogger(__name__)


def _int_to_base36(num: int) -> str:
    """
    Convert an integer to its base-36 representation using numbers and uppercase letters.

    Base-36 encoding uses digits 0-9 and uppercase letters A-Z, providing a case-insensitive
    alphanumeric representation. The function takes an integer `num` as input and returns
    its base-36 representation as a string.

    Parameters
    ----------
    num
        The integer to convert to base-36.

    Returns
    -------
    str
        The base-36 representation of the input integer.

    Examples
    --------
    >>> num = 123456
    >>> base36_num = int_to_base36(num)
    >>> print(base36_num)
    '2N9'
    """
    ALPHANUMERIC_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    if num == 0:
        return ALPHANUMERIC_CHARS[0]

    base36_str = ""
    base = len(ALPHANUMERIC_CHARS)

    while num:
        num, remainder = divmod(num, base)
        base36_str = ALPHANUMERIC_CHARS[remainder] + base36_str

    return base36_str


def _short_hash(string: str, length: int = 4) -> str:
    """Create a hash of 'string' with length 'length'."""
    return base64.b32encode(string.encode()).decode("utf-8").zfill(length)[:length]


def _remove_vowels(input_str: str, length: int = 4) -> str:
    vowels = "aeiouAEIOU"
    output_str = "".join([char for char in input_str if char not in vowels])
    return output_str.zfill(length)[:length]


def create_context(
    name: str, which: str, index: int, parent: Context | None = None
) -> Context:
    """Create a context that can identify this integration."""
    # Use a hash for the name because otherwise the context might become
    # too long (max len == 26) to fit in the database.
    # Pack index with base85 to maximize the number of contexts we can create
    # before we exceed the 26-character limit and are forced to wrap.
    time_stamp = ulid_transform.ulid_now()[:10]  # time part of a ULID
    name_hash = _short_hash(name)
    which_short = _remove_vowels(which)
    context_id_start = f"{time_stamp}:{_DOMAIN_SHORT}:{name_hash}:{which_short}:"
    chars_left = 26 - len(context_id_start)
    index_packed = _int_to_base36(index).zfill(chars_left)[-chars_left:]
    context_id = context_id_start + index_packed
    parent_id = parent.id if parent else None
    return Context(id=context_id, parent_id=parent_id)


def is_our_context(context: Context | None) -> bool:
    """Check whether this integration created 'context'."""
    if context is None:
        return False
    return f":{_DOMAIN_SHORT}:" in context.id


def _split_service_data(service_data, adapt_brightness, adapt_color):
    """Split service_data into two dictionaries (for color and brightness)."""
    transition = service_data.get(ATTR_TRANSITION)
    if transition is not None:
        # Split the transition over both commands
        service_data[ATTR_TRANSITION] /= 2
    service_datas = []
    if adapt_brightness:
        service_data_brightness = service_data.copy()
        service_data_brightness.pop(ATTR_RGB_COLOR, None)
        service_data_brightness.pop(ATTR_COLOR_TEMP_KELVIN, None)
        service_datas.append(service_data_brightness)
    if adapt_color:
        service_data_color = service_data.copy()
        service_data_color.pop(ATTR_BRIGHTNESS, None)
        service_datas.append(service_data_color)

    if not service_datas:  # neither adapt_brightness nor adapt_color
        return [service_data]
    return service_datas


def match_switch_state_event(event: Event, from_or_to_state: list[str]):
    """Match state event when either 'from_state' or 'to_state' matches."""
    old_state = event.data.get("old_state")
    from_state_match = old_state is not None and old_state.state in from_or_to_state

    new_state = event.data.get("new_state")
    to_state_match = new_state is not None and new_state.state in from_or_to_state

    match = from_state_match or to_state_match
    return match


def _expand_light_groups(hass: HomeAssistant, lights: list[str]) -> list[str]:
    all_lights = set()
    turn_on_off_listener = hass.data[DOMAIN][ATTR_TURN_ON_OFF_LISTENER]
    for light in lights:
        state = hass.states.get(light)
        if state is None:
            _LOGGER.debug("State of %s is None", light)
            all_lights.add(light)
        elif "entity_id" in state.attributes:  # it's a light group
            group = state.attributes["entity_id"]
            turn_on_off_listener.lights.discard(light)
            all_lights.update(group)
            _LOGGER.debug("Expanded %s to %s", light, group)
        else:
            all_lights.add(light)
    return list(all_lights)


def _supported_features(hass: HomeAssistant, light: str):
    state = hass.states.get(light)
    supported_features = state.attributes.get(ATTR_SUPPORTED_FEATURES, 0)
    supported = {
        key for key, value in _SUPPORT_OPTS.items() if supported_features & value
    }
    supported_color_modes = state.attributes.get(ATTR_SUPPORTED_COLOR_MODES, set())
    if COLOR_MODE_RGB in supported_color_modes:
        supported.add("color")
        # Adding brightness here, see
        # comment https://github.com/basnijholt/adaptive-lighting/issues/112#issuecomment-836944011
        supported.add("brightness")
    if COLOR_MODE_RGBW in supported_color_modes:
        supported.add("color")
        supported.add("brightness")  # see above url
    if COLOR_MODE_XY in supported_color_modes:
        supported.add("color")
        supported.add("brightness")  # see above url
    if COLOR_MODE_HS in supported_color_modes:
        supported.add("color")
        supported.add("brightness")  # see above url
    if COLOR_MODE_COLOR_TEMP in supported_color_modes:
        supported.add("color_temp")
        supported.add("brightness")  # see above url
    if COLOR_MODE_BRIGHTNESS in supported_color_modes:
        supported.add("brightness")
    return supported


def color_difference_redmean(
    rgb1: tuple[float, float, float], rgb2: tuple[float, float, float]
) -> float:
    """Distance between colors in RGB space (redmean metric).

    The maximal distance between (255, 255, 255) and (0, 0, 0) ≈ 765.

    Sources:
    - https://en.wikipedia.org/wiki/Color_difference#Euclidean
    - https://www.compuphase.com/cmetric.htm
    """
    r_hat = (rgb1[0] + rgb2[0]) / 2
    delta_r, delta_g, delta_b = ((col1 - col2) for col1, col2 in zip(rgb1, rgb2))
    red_term = (2 + r_hat / 256) * delta_r**2
    green_term = 4 * delta_g**2
    blue_term = (2 + (255 - r_hat) / 256) * delta_b**2
    return math.sqrt(red_term + green_term + blue_term)


# All comparisons should be done with RGB since
# converting anything to color temp is inaccurate.
def _convert_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    if ATTR_RGB_COLOR in attributes:
        return attributes

    rgb = None
    if ATTR_COLOR_TEMP_KELVIN in attributes:
        rgb = color_temperature_to_rgb(attributes[ATTR_COLOR_TEMP_KELVIN])
    elif ATTR_XY_COLOR in attributes:
        rgb = color_xy_to_RGB(*attributes[ATTR_XY_COLOR])

    if rgb is not None:
        attributes[ATTR_RGB_COLOR] = rgb
        _LOGGER.debug(f"Converted {attributes} to rgb {rgb}")
    else:
        _LOGGER.debug("No suitable conversion found")

    return attributes


def _add_missing_attributes(
    old_attributes: dict[str, Any],
    new_attributes: dict[str, Any],
) -> dict[str, Any]:
    if not any(
        attr in old_attributes and attr in new_attributes
        for attr in [ATTR_COLOR_TEMP_KELVIN, ATTR_RGB_COLOR]
    ):
        old_attributes = _convert_attributes(old_attributes)
        new_attributes = _convert_attributes(new_attributes)

    return old_attributes, new_attributes


def _attributes_have_changed(
    light: str,
    old_attributes: dict[str, Any],
    new_attributes: dict[str, Any],
    adapt_brightness: bool,
    adapt_color: bool,
    context: Context,
) -> bool:
    if adapt_color:
        old_attributes, new_attributes = _add_missing_attributes(
            old_attributes, new_attributes
        )

    if (
        adapt_brightness
        and ATTR_BRIGHTNESS in old_attributes
        and ATTR_BRIGHTNESS in new_attributes
    ):
        last_brightness = old_attributes[ATTR_BRIGHTNESS]
        current_brightness = new_attributes[ATTR_BRIGHTNESS]
        if abs(current_brightness - last_brightness) > BRIGHTNESS_CHANGE:
            _LOGGER.debug(
                "Brightness of '%s' significantly changed from %s to %s with"
                " context.id='%s'",
                light,
                last_brightness,
                current_brightness,
                context.id,
            )
            return True

    if (
        adapt_color
        and ATTR_COLOR_TEMP_KELVIN in old_attributes
        and ATTR_COLOR_TEMP_KELVIN in new_attributes
    ):
        last_color_temp = old_attributes[ATTR_COLOR_TEMP_KELVIN]
        current_color_temp = new_attributes[ATTR_COLOR_TEMP_KELVIN]
        if abs(current_color_temp - last_color_temp) > COLOR_TEMP_CHANGE:
            _LOGGER.debug(
                "Color temperature of '%s' significantly changed from %s to %s with"
                " context.id='%s'",
                light,
                last_color_temp,
                current_color_temp,
                context.id,
            )
            return True

    if (
        adapt_color
        and ATTR_RGB_COLOR in old_attributes
        and ATTR_RGB_COLOR in new_attributes
    ):
        last_rgb_color = old_attributes[ATTR_RGB_COLOR]
        current_rgb_color = new_attributes[ATTR_RGB_COLOR]
        redmean_change = color_difference_redmean(last_rgb_color, current_rgb_color)
        if redmean_change > RGB_REDMEAN_CHANGE:
            _LOGGER.debug(
                "color RGB of '%s' significantly changed from %s to %s with"
                " context.id='%s'",
                light,
                last_rgb_color,
                current_rgb_color,
                context.id,
            )
            return True
    return False


class _AsyncSingleShotTimer:
    def __init__(self, delay, callback):
        """Initialize the timer."""
        self.delay = delay
        self.callback = callback
        self.task = None
        self.start_time: int | None = None

    async def _run(self):
        """Run the timer. Don't call this directly, use start() instead."""
        self.start_time = dt_util.utcnow()
        await asyncio.sleep(self.delay)
        if self.callback:
            if asyncio.iscoroutinefunction(self.callback):
                await self.callback()
            else:
                self.callback()

    def is_running(self):
        """Return whether the timer is running."""
        return self.task is not None and not self.task.done()

    def start(self):
        """Start the timer."""
        if self.task is not None and not self.task.done():
            self.task.cancel()
        self.task = asyncio.create_task(self._run())

    def cancel(self):
        """Cancel the timer."""
        if self.task:
            self.task.cancel()
            self.callback = None

    def remaining_time(self):
        """Return the remaining time before the timer expires."""
        if self.start_time is not None:
            elapsed_time = (dt_util.utcnow() - self.start_time).total_seconds()
            return max(0, self.delay - elapsed_time)
        return 0


def is_light_on_off_event(event: Event) -> bool:
    if event.data.get(ATTR_DOMAIN) != LIGHT_DOMAIN:
        return False

    if event.data.get(ATTR_SERVICE) not in [SERVICE_TURN_ON, SERVICE_TURN_OFF]:
        return False

    return True


def get_entity_ids_from_service_event(hass: HomeAssistant, event: Event) -> list[str]:
    entity_ids = []

    service_data = event.data[ATTR_SERVICE_DATA]
    if ATTR_ENTITY_ID in service_data:
        entity_ids = cv.ensure_list_csv(service_data[ATTR_ENTITY_ID])
    elif ATTR_AREA_ID in service_data:
        area_ids = cv.ensure_list_csv(service_data[ATTR_AREA_ID])
        for area_id in area_ids:
            area_entity_ids = area_entities(hass, area_id)
            for entity_id in area_entity_ids:
                if entity_id.startswith(LIGHT_DOMAIN):
                    entity_ids.append(entity_id)
            _LOGGER.debug("Found entity_ids '%s' for area_id '%s'", entity_ids, area_id)
    else:
        _LOGGER.debug(
            "No entity_ids or area_ids found in service_data: %s", service_data
        )

    return entity_ids

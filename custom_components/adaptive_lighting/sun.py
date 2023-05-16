import astral
import logging
import bisect
from dataclasses import dataclass
import datetime
from datetime import timedelta
import math
from typing import Literal

import homeassistant.util.dt as dt_util
from homeassistant.util.color import (
    color_RGB_to_xy,
    color_temperature_to_rgb,
    color_xy_to_hs,
)
from homeassistant.const import SUN_EVENT_SUNRISE, SUN_EVENT_SUNSET

from .const import SUN_EVENT_MIDNIGHT, SUN_EVENT_NOON

_ORDER = (SUN_EVENT_SUNRISE, SUN_EVENT_NOON, SUN_EVENT_SUNSET, SUN_EVENT_MIDNIGHT)
_ALLOWED_ORDERS = {_ORDER[i:] + _ORDER[:i] for i in range(len(_ORDER))}

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SunLightSettings:
    """Track the state of the sun and associated light settings."""

    name: str
    astral_location: astral.Location
    adapt_until_sleep: bool
    max_brightness: int
    max_color_temp: int
    min_brightness: int
    min_color_temp: int
    sleep_brightness: int
    sleep_rgb_or_color_temp: Literal["color_temp", "rgb_color"]
    sleep_color_temp: int
    sleep_rgb_color: tuple[int, int, int]
    sunrise_offset: datetime.timedelta | None
    sunrise_time: datetime.time | None
    max_sunrise_time: datetime.time | None
    sunset_offset: datetime.timedelta | None
    sunset_time: datetime.time | None
    min_sunset_time: datetime.time | None
    time_zone: datetime.tzinfo
    transition: int

    def get_sun_events(self, date: datetime.datetime) -> dict[str, float]:
        """Get the four sun event's timestamps at 'date'."""

        def _replace_time(date: datetime.datetime, key: str) -> datetime.datetime:
            time = getattr(self, f"{key}_time")
            date_time = datetime.datetime.combine(date, time)
            try:  # HA ≤2021.05, https://github.com/basnijholt/adaptive-lighting/issues/128
                utc_time = self.time_zone.localize(date_time).astimezone(dt_util.UTC)
            except AttributeError:  # HA ≥2021.06
                utc_time = date_time.replace(
                    tzinfo=dt_util.DEFAULT_TIME_ZONE
                ).astimezone(dt_util.UTC)
            return utc_time

        def calculate_noon_and_midnight(
            sunset: datetime.datetime, sunrise: datetime.datetime
        ) -> tuple[datetime.datetime, datetime.datetime]:
            middle = abs(sunset - sunrise) / 2
            if sunset > sunrise:
                noon = sunrise + middle
                midnight = noon + timedelta(hours=12) * (1 if noon.hour < 12 else -1)
            else:
                midnight = sunset + middle
                noon = midnight + timedelta(hours=12) * (
                    1 if midnight.hour < 12 else -1
                )
            return noon, midnight

        location = self.astral_location

        sunrise = (
            location.sunrise(date, local=False)
            if self.sunrise_time is None
            else _replace_time(date, "sunrise")
        ) + self.sunrise_offset
        sunset = (
            location.sunset(date, local=False)
            if self.sunset_time is None
            else _replace_time(date, "sunset")
        ) + self.sunset_offset

        if self.max_sunrise_time is not None:
            max_sunrise = _replace_time(date, "max_sunrise")
            if max_sunrise < sunrise:
                sunrise = max_sunrise

        if self.min_sunset_time is not None:
            min_sunset = _replace_time(date, "min_sunset")
            if min_sunset > sunset:
                sunset = min_sunset

        if (
            self.sunrise_time is None
            and self.sunset_time is None
            and self.max_sunrise_time is None
            and self.min_sunset_time is None
        ):
            try:
                # Astral v1
                solar_noon = location.solar_noon(date, local=False)
                solar_midnight = location.solar_midnight(date, local=False)
            except AttributeError:
                # Astral v2
                solar_noon = location.noon(date, local=False)
                solar_midnight = location.midnight(date, local=False)
        else:
            (solar_noon, solar_midnight) = calculate_noon_and_midnight(sunset, sunrise)

        events = [
            (SUN_EVENT_SUNRISE, sunrise.timestamp()),
            (SUN_EVENT_SUNSET, sunset.timestamp()),
            (SUN_EVENT_NOON, solar_noon.timestamp()),
            (SUN_EVENT_MIDNIGHT, solar_midnight.timestamp()),
        ]
        # Check whether order is correct
        events = sorted(events, key=lambda x: x[1])
        events_names, _ = zip(*events)
        if events_names not in _ALLOWED_ORDERS:
            msg = (
                f"{self.name}: The sun events {events_names} are not in the expected"
                " order. The Adaptive Lighting integration will not work!"
                " This might happen if your sunrise/sunset offset is too large or"
                " your manually set sunrise/sunset time is past/before noon/midnight."
            )
            _LOGGER.error(msg)
            raise ValueError(msg)

        return events

    def _get_relevant_events(self, now: datetime.datetime) -> list[tuple[str, float]]:
        """Get the previous and next sun event."""
        events = [
            self.get_sun_events(now + timedelta(days=days)) for days in [-1, 0, 1]
        ]
        events = sum(events, [])  # flatten lists
        events = sorted(events, key=lambda x: x[1])
        i_now = bisect.bisect([ts for _, ts in events], now.timestamp())
        return events[i_now - 1 : i_now + 1]

    def _calc_percent(self, transition: int) -> float:
        """Calculate the position of the sun in %."""
        now = dt_util.utcnow()

        target_time = now + timedelta(seconds=transition)
        target_ts = target_time.timestamp()
        today = self._get_relevant_events(target_time)
        (_, prev_ts), (next_event, next_ts) = today
        h, x = (  # pylint: disable=invalid-name
            (prev_ts, next_ts)
            if next_event in (SUN_EVENT_SUNSET, SUN_EVENT_SUNRISE)
            else (next_ts, prev_ts)
        )
        k = 1 if next_event in (SUN_EVENT_SUNSET, SUN_EVENT_NOON) else -1
        percentage = (0 - k) * ((target_ts - h) / (h - x)) ** 2 + k
        return percentage

    def _calc_brightness_pct(self, percent: float, is_sleep: bool) -> float:
        """Calculate the brightness in %."""
        if is_sleep:
            return self.sleep_brightness
        if percent > 0:
            return self.max_brightness
        delta_brightness = self.max_brightness - self.min_brightness
        percent = 1 + percent
        return (delta_brightness * percent) + self.min_brightness

    def _calc_color_temp_kelvin(self, percent: float) -> int:
        """Calculate the color temperature in Kelvin."""
        if percent > 0:
            delta = self.max_color_temp - self.min_color_temp
            ct = (delta * percent) + self.min_color_temp
            return 5 * round(ct / 5)  # round to nearest 5
        if percent == 0 or not self.adapt_until_sleep:
            return self.min_color_temp
        if self.adapt_until_sleep and percent < 0:
            delta = abs(self.min_color_temp - self.sleep_color_temp)
            ct = (delta * abs(1 + percent)) + self.sleep_color_temp
            return 5 * round(ct / 5)  # round to nearest 5

    def calculate(
        self, is_sleep, transition
    ) -> dict[str, float | int | tuple[float, float] | tuple[float, float, float]]:
        """Get all light settings.

        Calculating all values takes <0.5ms.
        """
        percent = (
            self._calc_percent(transition)
            if transition is not None
            else self._calc_percent(0)
        )
        brightness_pct = self._calc_brightness_pct(percent, is_sleep)
        if is_sleep:
            color_temp_kelvin = self.sleep_color_temp
            rgb_color: tuple[float, float, float] = self.sleep_rgb_color
        else:
            color_temp_kelvin = self._calc_color_temp_kelvin(percent)
            rgb_color: tuple[float, float, float] = color_temperature_to_rgb(
                color_temp_kelvin
            )
        # backwards compatibility for versions < 1.3.1 - see #403
        color_temp_mired: float = math.floor(1000000 / color_temp_kelvin)
        xy_color: tuple[float, float] = color_RGB_to_xy(*rgb_color)
        hs_color: tuple[float, float] = color_xy_to_hs(*xy_color)
        return {
            "brightness_pct": brightness_pct,
            "color_temp_kelvin": color_temp_kelvin,
            "color_temp_mired": color_temp_mired,
            "rgb_color": rgb_color,
            "xy_color": xy_color,
            "hs_color": hs_color,
            "sun_position": percent,
        }

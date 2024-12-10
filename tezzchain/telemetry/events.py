"""
Telemetry events module for tracking usage and error metrics.

This module provides a telemetry system that can track events both in TezzChain's internal system
and optionally in a user's separate tracking system. It collects system metrics and handles
event tracking while respecting user privacy and telemetry preferences.

The telemetry can be disabled by setting the TEZZCHAIN_TELEMETRY environment variable to "false".
"""

import os
import json
import uuid
import logging
import platform
from typing import Optional, Literal

import psutil
from posthog import Posthog

from tezzchain import constants as const


logger = logging.getLogger("tezzchain")


class SemiAnonymizedTelemetry:
    """
    A telemetry system that tracks events both in TezzChain's system and optionally in a user's system.

    This class manages event tracking with system metrics collection, supporting both usage and error events.
    It generates and maintains a unique identifier for the TezzChain installation and can optionally
    track events in a separate user-provided PostHog instance.

    Args:
        api_key (Optional[str]): PostHog API key for user's tracking instance. If provided, host is required.
        host (Optional[str]): PostHog host URL for user's tracking instance.
        allow (Literal["ALL", "ERROR", "NONE"]): Telemetry permission level. Defaults to "ALL".
            - "ALL": Track both usage and error events
            - "ERROR": Track only error events
            - "NONE": Don't track any events

    Raises:
        ValueError: If api_key is provided without a host.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        allow: Literal["ALL", "ERROR", "NONE"] = "ALL",
    ):
        self.user_client = None
        if api_key is not None:
            if host is None:
                raise ValueError("Host is required if api_key is provided")
            self.user_client = Posthog(api_key, host=host)
        self.tezzchain_client = Posthog(
            const.TELEMETRY_API_KEY, host=const.TELEMETRY_HOST
        )
        self.allow = allow
        self.tezzchain_user_id = self.__generate_user_id()

    def __generate_user_id(self) -> str:
        """
        Generate a unique identifier for the TezzChain installation.

        If a configuration file exists, it loads the user ID from the file. Otherwise, it generates a new ID,
        saves it to the configuration file, and returns the new ID.
        """
        if const.TEZZCHAIN_CONFIG_FILE.exists():
            with open(const.TEZZCHAIN_CONFIG_FILE, "r") as f:
                config = json.load(f)
            return config.get("user_id", None)
        else:
            user_id = str(uuid.uuid4())
            with open(const.TEZZCHAIN_CONFIG_FILE, "w") as f:
                json.dump({"user_id": user_id}, f)
            return user_id

    def capture(
        self,
        event_name: str,
        properties: Optional[dict] = None,
        user_id: Optional[str] = None,
        user_properties: Optional[dict] = None,
        process_id: Optional[int] = os.getpid(),
        event_type: Literal["usage", "error"] = "usage",
    ):
        """
        Capture an event with system metrics in TezzChain's telemetry system and optionally in a user's system.

        This method captures events with automatically collected system metrics including OS details,
        Python version, CPU usage, memory usage, and thread count. Events can be captured in both
        TezzChain's telemetry system and optionally in a user's PostHog instance if configured.

        Args:
            event_name (str): Name of the event to capture.
            properties (Optional[dict]): Properties that will be captured to both accounts.
            user_id (Optional[str]): User ID for tracking events in the user's PostHog instance.
            user_properties (Optional[dict]): Properties that will only be captured to the user's account.
            process_id (Optional[int]): Process ID for collecting system metrics. Defaults to the current process ID.
            event_type (Literal["usage", "error"]): Type of event to capture. Defaults to "usage".

        Notes:
            - If event_type is "usage" and self.allow is not "ALL", the event will not be captured.
            - If event_type is "error" and self.allow is "NONE", the event will not be captured.
            - System metrics are automatically collected and added to the properties.
            - Failed event captures are logged but do not raise exceptions.
        """
        if event_type == "usage" and self.allow != "ALL":
            return  # User has not permitted telemetry for their application for recording usage metrics
        if event_type == "error" and self.allow == "NONE":
            return  # User has not permitted telemetry for their application for recording errors
        if properties is None:
            properties = {}
        custom_properties = {
            "operating_system": platform.system(),
            "tezzchain_version": const.TEZZCHAIN_VERSION,
            "python_version": platform.python_version(),
            "architecture": platform.machine(),
            "cpu_count": os.cpu_count(),
            "memory": psutil.virtual_memory().total,
            "memory_usage": psutil.Process(process_id).memory_info().rss,
            "num_threads": psutil.Process(process_id).num_threads(),
            "cpu_usage": psutil.Process(process_id).cpu_percent(interval=1),
        }
        properties = {**custom_properties, **properties}
        try:
            self.tezzchain_client.capture(
                self.tezzchain_user_id, event_name, properties
            )
        except Exception as e:
            logger.exception(f"Failed to capture event {event_name} to tezzchain: {e}")
        if self.user_client is not None and user_id is not None:
            if user_properties is None:
                user_properties = {}
            user_properties = {**user_properties, **properties}
            try:
                self.user_client.capture(user_id, event_name, user_properties)
            except Exception as e:
                logger.exception(f"Failed to capture event {event_name} to user: {e}")

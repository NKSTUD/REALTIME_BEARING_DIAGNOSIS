from dataclasses import asdict
from typing import Dict, List

import streamlit as st
from nidaqmx.system import System

from data_processor import Sensor

system = System.local()

DEFAULT_SENSORS = {
    "cDAQ1Mod1": ["cDAQ1Mod1/ai0", "cDAQ1Mod1/ai1", "cDAQ1Mod1/ai2", "cDAQ1Mod1/neutral"],
    "cDAQ1Mod2": ["cDAQ1Mod2/ai0", "cDAQ1Mod2/ai1", "cDAQ1Mod2/ai2", "cDAQ1Mod2/ai3"],
    "cDAQ1Mod3": ["cDAQ1Mod3/ai0", "cDAQ1Mod3/ai1", "cDAQ1Mod3/ai2", "cDAQ1Mod3/ai3"]
}


# def get_devices_and_channels() -> Dict[str, List[str]]:
#     """Return a dictionary of devices and their channels."""
#
#     return {device: system.devices[device].ai_physical_chans.channel_names for device in
#             system.devices} or DEFAULT_SENSORS

def get_devices_and_channels() -> Dict[str, List[str]]:
    """Return a dictionary of devices and their channels."""
    return {device.name: device.ai_physical_chans.channel_names for device in
            system.devices} or DEFAULT_SENSORS


def configure_sensor(sensor: str) -> Sensor:
    """Configure a single sensor and return a Sensor instance."""
    st.write(f"Sensor {sensor} Configuration")
    cols = st.columns(5)

    return Sensor(
        physical_channel=sensor,
        name=cols[4].text_input("Name", value=sensor, key=f"{sensor}_name"),
        sensitivity=cols[1].number_input("Sensitivity", value=1.0, key=f"{sensor}_sensitivity"),
        min_value=cols[2].number_input("Min Value", value=-5.0, key=f"{sensor}_min_value"),
        max_value=cols[3].number_input("Max Value", value=5.0, key=f"{sensor}_max_value"),
        type=cols[0].selectbox("Sensor Type", ["accelerometer", "microphone"], index=0, key=f"{sensor}_type")
    )


def sensor_configuration() -> Dict[str, List[Sensor]]:
    """Configure sensors for selected devices and return a dictionary of Sensor instances."""
    devices_and_channels = get_devices_and_channels()
    selected_devices = st.multiselect("Which Modules are you using?", list(devices_and_channels.keys()), key="device")

    selected_sensors = {}
    for device in selected_devices:
        channels = devices_and_channels[device]
        selected_channels = st.multiselect(f"Which sensors of {device} are you using?", channels, key=device)

        selected_sensors[device] = [configure_sensor(sensor) for sensor in selected_channels]

    return selected_sensors


def format_sensors_for_get_data(sensors: Dict[str, List[Sensor]]) -> List[dict]:
    """Format sensors for get_data function."""
    return [asdict(sensor) for device_sensors in sensors.values() for sensor in device_sensors]

from dataclasses import dataclass, asdict
from typing import List, Optional

import nidaqmx
import numpy as np
import pandas as pd
from nidaqmx.constants import AcquisitionType


@dataclass
class Sensor:
    physical_channel: str
    name: str
    sensitivity: float = 1.0
    min_value: float = -5.0
    max_value: float = 5.0
    type: str = "accelerometer"


def default_sensors() -> List[dict]:
    sensors = [
        Sensor("cDAQ1Mod3/ai0", "vertical_accelerometer_1", 1.0, -5, 5),
        Sensor("cDAQ1Mod3/ai1", "vertical_accelerometer_2", 1.0, -5, 5),
        Sensor("cDAQ1Mod3/ai2", "horizontal_accelerometer", 1.0, -5, 5),
        Sensor("cDAQ1Mod3/ai3", "microphone", 1.0, -5, 5, type='microphone'), ]
    return [asdict(sensor) for sensor in sensors]


def get_data(
        sensors: Optional[List[dict]] = None,
        sampling_rate: int = 25600,
        num_samples: int = 25600,
        is_started: bool = False,
        session_data: Optional[List[List[float]]] = None, is_simulation: bool = False
) -> pd.DataFrame:
    """
    Get data from sensors or return provided session data.

    :param is_simulation:
    :param sensors: List of sensor dictionaries. If None, uses default sensors.
    :param sampling_rate: Sampling rate in Hz.
    :param num_samples: Number of samples to acquire per channel.
    :param is_started: Whether to start the task and acquire new data.
    :param session_data: Existing data to use if is_started is False.
    :return: DataFrame containing acquired or provided data.
    """
    if sensors is None:
        sensors = default_sensors()

    sensor_names = [sensor['name'] for sensor in sensors]

    is_started = is_started and not is_simulation

    if is_started:
        try:
            with nidaqmx.Task() as task:
                for sensor in sensors:
                    if not sensor['type'] == 'accelerometer':
                        task.ai_channels.add_ai_microphone_chan(
                            physical_channel=sensor['physical_channel'],
                            mic_sensitivity=sensor['sensitivity']
                        )
                    else:
                        task.ai_channels.add_ai_accel_chan(
                            physical_channel=sensor['physical_channel'],
                            min_val=sensor['min_value'],
                            max_val=sensor['max_value'],
                            sensitivity=sensor['sensitivity']
                        )
                task.timing.cfg_samp_clk_timing(
                    sampling_rate,
                    samps_per_chan=num_samples,
                    sample_mode=AcquisitionType.CONTINUOUS
                )
                task.start()



                data = task.read(number_of_samples_per_channel=num_samples, timeout=10.0)

                return pd.DataFrame(data).T.set_axis(sensor_names, axis=1) if isinstance(data[0],
                                                                                         list) else pd.DataFrame(data,
                                                                                     columns=sensor_names)
        except nidaqmx.DaqError as e:
            print(f"Erreur lors de l'acquisition des données: {e}")
            return pd.DataFrame(columns=sensor_names)
    elif is_simulation:
        simulated_data = np.random.randn(num_samples, len(sensors))
        return pd.DataFrame(simulated_data, columns=sensor_names) if session_data is None else pd.DataFrame(
            session_data, columns=sensor_names)
    else:
        return pd.DataFrame(session_data, columns=sensor_names) if session_data is not None else pd.DataFrame(
            columns=sensor_names)


if __name__ == "__main__":
    data = get_data(is_started=True)
    print(data)

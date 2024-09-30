import threading
import time
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


class DataAcquisitionThread(threading.Thread):
    def __init__(self, data_queue, stop_event, sampling_rate, number_of_samples, selected_sensors, simulation):
        threading.Thread.__init__(self)
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.sampling_rate = sampling_rate
        self.number_of_samples = number_of_samples
        self.selected_sensors = selected_sensors
        self.simulation = simulation

    def run(self):
        while not self.stop_event.is_set():
            data = get_data(
                sensors=self.selected_sensors,
                sampling_rate=self.sampling_rate,
                num_samples=self.number_of_samples,
                is_started=True,
                is_simulation=self.simulation
            )
            self.data_queue.put(data)
            time.sleep(0.1)

if __name__ == "__main__":
    from queue import Queue

    data_queue = Queue()
    stop_event = threading.Event()

    acquisition_thread = DataAcquisitionThread(
        data_queue=data_queue,
        stop_event=stop_event,
        sampling_rate=25600,
        number_of_samples=25600,
        selected_sensors=default_sensors(),
        simulation=False
    )
    acquisition_thread.start()

    try:
        while True:
            if not data_queue.empty():
                data = data_queue.get()
                print(data)
            time.sleep(0.5)  # Adjust sleep time as needed
    except KeyboardInterrupt:
        stop_event.set()
        acquisition_thread.join()
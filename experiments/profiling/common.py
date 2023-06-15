import time
import platform
import pynvml
from memory_profiler import memory_usage


BATCH_SIZES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
SAMPLE_RATE = 44100
SOUND_DURATION = 4.0
N_BATCHES = 100
N_ROUNDS = 10


timer = time.perf_counter


def auto_device():
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            return "gpu"
        else:
            return "cpu"
    except pynvml.NVMLError:
        return "cpu"


def init_mem(device: str):
    if device == "gpu":
        pynvml.nvmlInit()
    elif device == "cpu":
        pass
    else:
        raise ValueError("Invalid device. Must be \"cpu\" or \"gpu\"")


def measure_memory(device: str):
    if device == "gpu":
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo_initial = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return meminfo_initial.used
    elif device == "cpu":
        mem_initial = memory_usage(-1, interval=0.1, timeout=1)[0]
        return mem_initial
    else:
        raise ValueError("Invalid device. Must be \"cpu\" or \"gpu\"")


def cleanup_mem(device: str):
    if device == "gpu":
        pynvml.nvmlShutdown()
    elif device == "cpu":
        pass
    else:
        raise ValueError("Invalid device. Must be \"cpu\" or \"gpu\"")

def get_devicetype(device: str):
    return platform.processor() if device == "cpu" else pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))

import pandas
from tqdm.auto import tqdm
from torchsynth.config import SynthConfig
from torchsynth.synth import Voice
from common import timer, auto_device, init_mem, measure_memory, cleanup_mem, get_devicetype, BATCH_SIZES, SAMPLE_RATE, SOUND_DURATION, N_BATCHES, N_ROUNDS


def main():
    results = []
    device = auto_device()
    device_t = "cuda:0" if device == "gpu" else "cpu"

    init_mem(device)

    for batch_size in tqdm(BATCH_SIZES):
        synth_cfg = SynthConfig(
            batch_size=batch_size,
            sample_rate=SAMPLE_RATE,
            buffer_size_seconds=SOUND_DURATION,
            reproducible=False
        )

        synth_cfg.to(device_t)
        synth = Voice(synth_cfg).to(device_t)
        synth() # Initial run; we don't count this

        for k in tqdm(range(N_ROUNDS), leave=False):
            m_pre = measure_memory(device)
            t = timer()
            for _ in tqdm(range(N_BATCHES), leave=False):
                synth.randomize()
                _ = synth()
            t = timer() - t
            m_post = measure_memory(device)

            results.append({
                "batch_size": batch_size,
                "num_run": k,
                "time": t,
                "memory_pre": m_pre,
                "memory_post": m_post
            })

    cleanup_mem(device)

    df = pandas.DataFrame(results)
    df["n_batches"] = N_BATCHES
    df["device"] = device
    df["sample_rate"] = SAMPLE_RATE
    df["n_rounds"] = N_ROUNDS
    df["sound_duration"] = SOUND_DURATION
    df["device_type"] = get_devicetype(device)

    df.to_csv("torchsynth_%s.csv" % device, index=False)


if __name__ == "__main__":
    main()
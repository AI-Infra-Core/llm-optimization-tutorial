# src file: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/profiling.py

import contextlib
import os
import time
import torch
import pickle
from tools.logging import logger

# the number of warmup steps before the active step in each profiling cycle
WARMUP = 3

# how much memory allocation/free ops to record in memory snapshots
MEMORY_SNAPSHOT_MAX_ENTRIES = 100000

def get_trace_dir(profiling_config):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    trace_dir = os.path.join(profiling_config.save_folder, f"rank{rank}")
    os.makedirs(trace_dir, exist_ok=True)
    return trace_dir

@contextlib.contextmanager
def maybe_enable_profiling(profiling_config):
    # get user defined profiler settings

    if profiling_config.enable_profiling:
        trace_dir = get_trace_dir(profiling_config)
        
        logger.info(f"Profiling active. Traces will be saved at {trace_dir}. Profile steps: [{profiling_config.start_step}, {profiling_config.end_step})")
        
        active = profiling_config.end_step - profiling_config.start_step
        warmup = WARMUP
        wait = profiling_config.start_step - warmup
        assert (
            wait >= 0
        ), "profile_freq must be greater than or equal to warmup + active"

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir, f"{profiling_config.start_step}_{profiling_config.end_step}_iteration"),
            record_shapes=True,
            with_stack=True,
        ) as torch_profiler:
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


@contextlib.contextmanager
def maybe_enable_memory_snapshot(profiling_config):
    enable_snapshot = profiling_config.enable_memory_snapshot
    if enable_snapshot:
        snapshot_dir = get_trace_dir(profiling_config)

        class MemoryProfiler:
            def __init__(self, start_step: int, end_step: int):
                torch.cuda.memory._record_memory_history(
                    max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES
                )
                def oom_observer(device, alloc, device_alloc, device_free):
                    # snapshot right after an OOM happened
                    print('saving allocated state during OOM')
                    snapshot = torch.cuda.memory._snapshot()
                    with open(os.path.join(snapshot_dir, f"oom_rank.pickle"), 'wb') as f:
                        pickle.dump(snapshot, f)
                torch._C._cuda_attach_out_of_memory_observer(oom_observer)
                self.step_num = -1
                
                self.start_step = start_step
                self.end_step = end_step

            def step(self):
                self.step_num += 1
                if self.step_num >= self.start_step and self.step_num < self.end_step:
                    curr_snapshot_dir = snapshot_dir
                    logger.info(f"Dumping memory snapshot at step {self.step_num}")
                    begin = time.monotonic()
                    output_file = os.path.join(
                        curr_snapshot_dir, f"step{self.step_num}_memory_snapshot.pickle"
                    )
                    with open(output_file, "wb") as output:
                        pickle.dump(torch.cuda.memory._snapshot(), output)
                    logger.info(
                        f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds"
                    )

        logger.info(f"Memory profiler active. Snapshot will be saved at {snapshot_dir}")
        profiler = MemoryProfiler(profiling_config.start_step, profiling_config.end_step)
        yield profiler
    else:
        yield None
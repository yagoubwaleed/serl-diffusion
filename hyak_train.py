import time
import submitit
from diffusion_policy.configs import DiffusionModelRunConfig
import os
from train_script import run_training


SWEEPS = [(10, 900),
          (20, 700),
          (30, 600),
          (40, 600),
          (50, 500),
          (60, 500),
          (70, 300),
          (80, 300),
          (90, 200),
          (100, 200)]
save_folder = "outputs"

def main():
    print("creating executor")
    jobs = []
    executor = submitit.AutoExecutor(folder="slurm_log")
    # set timeout in min, and partition for running the job
    executor.update_parameters(slurm_partition="gpu-a40",
                               slurm_account="weirdlab",
                               slurm_name="experiment",
                               timeout_min=24*60,
                               mem_gb=1000,
                               slurm_gpus_per_node=1,
                               slurm_gpus_per_task=1,
                               slurm_ntasks_per_node=1,
                               )
    executor.update_parameters(slurm_array_parallelism=6)
    current_dir = os.getcwd()

    with executor.batch():
        # In here submit jobs, and add them to the list, but they are all considered to be batched.
        for num_trajectories, num_epochs in SWEEPS:
            config = DiffusionModelRunConfig(
                hydra = None,
                with_state = False,
                num_trajs = num_trajectories,
                checkpoint_path = f"{current_dir}/{save_folder}/checkpoint_w_{num_trajectories}_trajectories.pt",
                dataset_path = f"{current_dir}/peg_insert_100_demos_2024-02-11_13-35-54.pkl",
                num_epochs = num_epochs,
            )
            job = executor.submit(run_training, config)
            jobs.append(job)

    while jobs:
        done = []
        for i in range(len(jobs)):
            job = jobs[i]
            if job.done():
                done.append(i)
                print("Finished a job!")
        for i in reversed(done):
            del jobs[i]


if __name__ == "__main__":
    main()

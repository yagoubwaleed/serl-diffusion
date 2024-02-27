import time
import submitit
from diffusion_policy.configs import DiffusionModelRunConfig
import os
from train_script import run_training


def create_config(num_trajectories: int, num_epochs: int, save_folder):
    config = DiffusionModelRunConfig()
    config.hydra = None
    current_dir = os.getcwd()
    config.with_state = False
    config.num_trajs = num_trajectories
    config.checkpoint_path = f"{current_dir}/{save_folder}/checkpoint_w_{num_trajectories}_trajectories.pt"
    config.dataset_path = f"{current_dir}/peg_insert_100_demos_2024-02-11_13-35-54.pkl"
    config.num_epochs = num_epochs
    if not os.path.exists(f"{current_dir}/{save_folder}"):
        os.makedirs(f"{current_dir}/{save_folder}")
    return config


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
    with executor.batch():
        # In here submit jobs, and add them to the list, but they are all considered to be batched.
for num_trajectories, num_epochs in SWEEPS:
            config = create_config(num_trajectories, num_epochs, "outputs")
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

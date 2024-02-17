# Franka Diffusion:

Website for serl is https://serl-robot.github.io/
Website for diffusion policy is https://diffusion-policy.cs.columbia.edu/

## Installation
1. Conda Environment:
    create an environment with
    ```bash
    conda create -n serl python=3.10
    ```

2.  Install the [serl_franka_controller](https://github.com/rail-berkeley/serl_franka_controller) in order to run on the real robot
3.  pip install the `serl_robot_infra` folder
4.  in the root folder, run `pip install -r requirements.txt`

## Structure
The structure of the repo is as follows:
```bash
├── eval_script.py # Contains the code for the evaluation of trained policy
├── record_demos.py # Contains the code for recording demonstrations with a space mouse
├── diffusion_policy
│   ├── dataset.py # Contains the code for the dataste generated from serl demonstrations
│   ├── hydra_utils.py # Contains the code for the hydra utils
│   ├── networks.py # Contains the code for the networks
├── serl_robot_infra
│   ├── franka_env
│   │   ├── camera # Contains realsense code
│   │   └── envs
│   │       ├── franka_env.py # Base franka Environment
│   │       └── peg_env 
│   │           ├── config.py # Config for peg insertion 
│   │           └── franka_peg_insert.py # peg insertion environment
│   ├── robot_servers
│       └── franka_server.py # this is the server you run to connect to the robot
└── train_script.py # Script containing the training code. It is environment agnostic
```
## Example Usage:
1. Run the robot server potentially modifying the default flags. (make sure the franka has FCI activated)
    ```bash
    python serl_robot_infra/robot_servers/franka_server.py
    ```
2. Record demonstrations with the space mouse. Change the variable success_needed to modify the number of demonstrations you will collect
    ```bash
    python record_demos.py
    ```
3. Train the policy. You will have to modify the hydra configuration dataclass at the top of the script to have the right paths
    ```bash
    python train_script.py
    ```
4. Evaluate the policy. You will have to modify the hydra configuration dataclass at the top of the script to have the right paths
    ```bash
    python eval_script.py
    ```

## Todos:
- Maybe add octo?
- Make the eval script read the paramaters from the checkpoint
- Add vr teleop to the codebase
- Add the ability to randomly pick trajectories
- Add the ability to pause when demoing

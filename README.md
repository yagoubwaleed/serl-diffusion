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

## Todos:
- Maybe add octo?
- Make the eval script read the paramaters from the checkpoint
- Add vr teleop to the codebase
- Add the ability to randomly pick trajectories

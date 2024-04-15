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
├── data
│   ├── replay_dataset.py # Convert from the hdf5 format to usable data format
├── diffusion_policy
│   ├── dataset.py # Contains the code for the dataste generated from serl demonstrations
│   ├── configs.py # Contains the code for configs and hydra
│   ├── networks.py # Contains the code for the networks
│   ├── make_networks.py # Contains code to instantiate the training objects
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
├── eval_script.py # Contains the code for the evaluation of trained policy in the real world
├── record_demos.py # Contains the code for recording demonstrations with a space mouse
├── sim_eval.py # Code to evaluate on a robosuite environment
├── hyak_train.py # Code to run sweaps on a slurm cluster training multiple models
└── train_script.py # Script containing the training code. It is environment-agnostic
```
## Example Usage:
1. In a separate terminal,run the robot server, potentially modifying the default flags. (make sure the franka has FCI activated)
    ```bash
    killall -9 rosmaster
    python serl_robot_infra/robot_servers/franka_server.py
    ```
    You might get a jacobian error. This is fine. The gripper might also be very weird spewing error messages. In this case, close and reopen your terminal

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
## Training using robosuite:
1. Download the data that you want to use. To do this, you need to have cloned and installed the robomimic environment.
You can find the instructions [here](https://robomimic.github.io/docs/introduction/getting_started.html) which will eventually 
have you run a command like this `python robomimic/scripts/download_datasets.py --tasks lift --dataset_types ph`
2. You now need to create a json file that contains the environment information and preprocess it. To do this, run some version
of the following command documented on robomimic: 
   ```bash
   python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84
   ```
   this should output a json to the terminal which you can coppy directly. You do not need to finish the command. The paramaters
   in the json that I change are the camera names, as well as the camera size.
3. Process the data into the format that is usable in the serl-diffusion repo using the replay_dataset.py script. Make sure
   to update the dataset_config in the configs.py file to match the keys you want

4. You can now train the policy using the training script. Make sure to specify the correct data paths as well as the correct format
   (the jacob format)
5. Once you have a trained checkpoint, modify the eval config in the sim_eval.py file to point to the correct checkpoint
and then you can run eval.
## Todos:
~~- Add documentation + clean procedure on how to use robosuite~~
~~- Make the eval script read the paramaters from the checkpoint~~
- Add vr teleop to the codebase
- Add the ability to randomly pick trajectories
- Add the ability to pause when demoing 
- Add capability to checkpoint + resume training
- Add r3m to the vision encoder as an option

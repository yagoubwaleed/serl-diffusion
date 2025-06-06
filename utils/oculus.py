import threading
import time

import numpy as np
from oculus_reader.reader import OculusReader


def run_threaded_command(command, args=(), daemon=True):
    thread = threading.Thread(target=command, args=args, daemon=daemon)
    thread.start()

    return thread


from utils.transformations import (add_angles, euler_to_quat, quat_diff,
                                   quat_to_euler, rmat_to_quat)


def vec_to_reorder_mat(vec):
    X = np.zeros((len(vec), len(vec)))
    for i in range(X.shape[0]):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X


class VRController:
    def __init__(
            self,
            right_controller: bool = True,
            max_lin_vel: float = 1,
            max_rot_vel: float = 1,
            max_gripper_vel: float = 1,
            spatial_coeff: float = 1,
            pos_action_gain: float = 5,
            rot_action_gain: float = 2,
            gripper_action_gain: float = 3,
            rmat_reorder: list = [-2, -1, -3, 4],
    ):
        self.oculus_reader = OculusReader()
        self.vr_to_global_mat = np.eye(4)
        self.max_lin_vel = max_lin_vel
        self.max_rot_vel = max_rot_vel
        self.max_gripper_vel = max_gripper_vel
        self.spatial_coeff = spatial_coeff
        self.pos_action_gain = pos_action_gain
        self.rot_action_gain = rot_action_gain
        self.gripper_action_gain = gripper_action_gain
        self.global_to_env_mat = vec_to_reorder_mat(rmat_reorder)
        self.controller_id = "r" if right_controller else "l"
        self.reset_orientation = True
        self.reset_state()
        self._kill_thread = False
        # Start State Listening Thread #
        self._controller_thread = run_threaded_command(self._update_internal_state)

    def __del__(self):
        self._kill_thread = True
        self._controller_thread.join()
        self.oculus_reader.stop()

    def reset_state(self):
        self._state = {
            "poses": {},
            "buttons": {"A": False, "B": False},
            "movement_enabled": False,
            "controller_on": True,
        }
        self.update_sensor = True
        self.reset_origin = True
        self.robot_origin = None
        self.vr_origin = None
        self.vr_state = None

    def _update_internal_state(self, num_wait_sec=5, hz=50):
        last_read_time = time.time()
        while True:
            if self._kill_thread:
                break
            # Regulate Read Frequency #
            time.sleep(1 / hz)

            # Read Controller
            time_since_read = time.time() - last_read_time
            poses, buttons = self.oculus_reader.get_transformations_and_buttons()
            self._state["controller_on"] = time_since_read < num_wait_sec
            if poses == {}:
                continue

            # Determine Control Pipeline #
            toggled = self._state["movement_enabled"] != buttons["RG"]
            self.update_sensor = self.update_sensor or buttons["RG"]
            self.reset_orientation = self.reset_orientation or buttons["RJ"]
            self.reset_origin = self.reset_origin or toggled

            # Save Info #
            self._state["poses"] = poses
            self._state["buttons"] = buttons
            self._state["movement_enabled"] = buttons["RG"]
            self._state["controller_on"] = True
            last_read_time = time.time()

            # Update Definition Of "Forward" #
            stop_updating = (
                    self._state["buttons"]["RJ"] or self._state["movement_enabled"]
            )
            if self.reset_orientation:
                rot_mat = np.asarray(self._state["poses"][self.controller_id])
                if stop_updating:
                    self.reset_orientation = False
                # try to invert the rotation matrix, if not possible, then just use the identity matrix
                try:
                    rot_mat = np.linalg.inv(rot_mat)
                except:
                    print(f"exception for rot mat: {rot_mat}")
                    rot_mat = np.eye(4)
                    self.reset_orientation = True
                self.vr_to_global_mat = rot_mat

    def _process_reading(self):
        rot_mat = np.asarray(self._state["poses"][self.controller_id])
        rot_mat = self.global_to_env_mat @ self.vr_to_global_mat @ rot_mat
        vr_pos = self.spatial_coeff * rot_mat[:3, 3]
        vr_quat = rmat_to_quat(rot_mat[:3, :3])
        vr_gripper = self._state["buttons"]["rightTrig"][0]

        self.vr_state = {"pos": vr_pos, "quat": vr_quat, "gripper": vr_gripper}

    def _limit_velocity(self, lin_vel, rot_vel, gripper_vel):
        """Scales down the linear and angular magnitudes of the action"""
        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)
        gripper_vel_norm = np.linalg.norm(gripper_vel)
        if lin_vel_norm > self.max_lin_vel:
            lin_vel = lin_vel * self.max_lin_vel / lin_vel_norm
        if rot_vel_norm > self.max_rot_vel:
            rot_vel = rot_vel * self.max_rot_vel / rot_vel_norm
        if gripper_vel_norm > self.max_gripper_vel:
            gripper_vel = gripper_vel * self.max_gripper_vel / gripper_vel_norm
        return lin_vel, rot_vel, gripper_vel

    def _calculate_action(self, state_dict, include_info=False):
        # Read Sensor #
        if self.update_sensor:
            self._process_reading()
            self.update_sensor = False

        # Read Observation
        robot_pos = np.array(state_dict["tcp_pose"][:3])
        robot_euler = state_dict["tcp_pose"][3:]
        robot_quat = euler_to_quat(robot_euler)
        robot_gripper = state_dict["gripper_pose"]

        # Reset Origin On Release #
        if self.reset_origin:
            self.robot_origin = {"pos": robot_pos, "quat": robot_quat}
            self.vr_origin = {
                "pos": self.vr_state["pos"],
                "quat": self.vr_state["quat"],
            }
            self.reset_origin = False

        # Calculate Positional Action #
        robot_pos_offset = robot_pos - self.robot_origin["pos"]
        target_pos_offset = self.vr_state["pos"] - self.vr_origin["pos"]
        pos_action = target_pos_offset - robot_pos_offset

        # Calculate Euler Action #
        robot_quat_offset = quat_diff(robot_quat, self.robot_origin["quat"])
        target_quat_offset = quat_diff(self.vr_state["quat"], self.vr_origin["quat"])
        quat_action = quat_diff(target_quat_offset, robot_quat_offset)
        euler_action = quat_to_euler(quat_action)

        # Calculate Gripper Action #
        gripper_action = 1.0 if self.vr_state["gripper"] < 0.5 else -1.0


        # Calculate Desired Pose #
        target_pos = pos_action + robot_pos
        target_euler = add_angles(euler_action, robot_euler)
        target_cartesian = np.concatenate([target_pos, target_euler])

        # Scale Appropriately #
        pos_action *= self.pos_action_gain
        euler_action *= self.rot_action_gain
        # gripper_action *= self.gripper_action_gain
        lin_vel, rot_vel, gripper_vel = self._limit_velocity(
            pos_action, euler_action, gripper_action
        )
        # Prepare Return Values #
        info_dict = {
            "target_cartesian_position": target_cartesian,
            "target_gripper_position": gripper_action,
        }
        # action = np.concatenate([lin_vel, rot_vel, [gripper_vel]])
        action = np.concatenate([pos_action, euler_action, [gripper_action]])

        action = np.clip(action, -1, 1)
        # Return #
        if include_info:
            return action, info_dict
        else:
            return action

    def get_info(self):
        return {
            "save_demo": self._state["buttons"]["A"],
            "discard_demo": self._state["buttons"]["B"],
            "movement_enabled": self._state["movement_enabled"],
            "controller_on": self._state["controller_on"],
        }

    def forward(self, obs_dict, DoF=6, include_info=False):
        if self._state["poses"] == {}:
            action = np.zeros(DoF + 1)
            if include_info:
                return action, {}
            else:
                return action

        return self._calculate_action(
            obs_dict["state"], include_info=include_info
        )

    def save_demo(self) -> bool:
        print("Do you want to save the demo? Press A to save the demo and B to discard the demo")

        # Loop until the user enters what they want to do and then return it
        while True:
            info_dict = self.get_info()
            if info_dict["save_demo"]:
                return True
            elif info_dict["discard_demo"]:
                return False
            else:
                time.sleep(0.1)

    def check_done(self) -> bool:
        info_dict = self.get_info()
        return info_dict["save_demo"] or info_dict["discard_demo"]



def main():
    oculus_reader = VRController()

    while True:
        time.sleep(0.3)
        print(oculus_reader.get_info())


if __name__ == "__main__":
    main()

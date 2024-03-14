import mujoco
import numpy as np
from typing import Dict

DEBUG_MODE = True


def quaternion_error(desired_quat: np.ndarray, current_quat: np.ndarray,
                     ) -> np.ndarray:
    """ Calculates the orientation error between two quaternions. """

    quat_conj = np.zeros(4,)
    mujoco.mju_negQuat(quat_conj, current_quat)
    quat_conj /= np.linalg.norm(quat_conj)

    quat_err = np.zeros(4,)
    mujoco.mju_mulQuat(quat_err, desired_quat, quat_conj)

    return quat_err[1:] * np.sign(quat_err[0])


class OSController:
    def __init__(self, model, data, config, eef_name="EEF"):
        """ Operational Space (OS) controller for the Franka Emika Panda robot
            (also known as cartesian impedance controller).

            The input of the controller are a target cartesian position and
            quaternion orientation based on the world frame.

            The controller will output torques in the joint space to achieve
            the desired end-effector configuration.

            Ref: [mujoco_controllers]
                 (https://github.com/peterdavidfagan/mujoco_controllers)
                 [mjctrl](https://github.com/kevinzakka/mjctrl/tree/main)

                 (https://studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/)
                 (https://studywolf.wordpress.com/2013/09/17/robot-control-5-controlling-in-the-null-space/)

            Args:
                model (MjModel): Mujoco model structure.
                data   (MjData): Mujoco data structure.
        """
        self.model, self.data = model, data
        self.eef_id = self.model.site(eef_name).id
        self.joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ['joint1', 'joint2', 'joint3', 'joint4', 'joint5',
                         'joint6', 'joint7']]

        self.target_vel, self.target_angular_vel = np.zeros(3), np.zeros(3)
        self.nullspace_joint_config = {
            "qpos": config["null"]["qpos"],
            "vel": np.zeros(len(self.joint_ids))}

        # Gains
        # NOTE: kd = 2 * np.sqrt(kp) * damping_ratio
        self.controller_gains = config["gains"]
        self.last_control_signal = np.zeros(len(self.joint_ids))

    def _get_eef_achieved_pose(self):
        """ Get current end-effecotr achieved (actual) pose in the world frame.
            # CHECKED: The order of quanterion is (w, x, y, z) in MuJoCo.
        """

        current_eef_pos = self.data.site_xpos[self.eef_id].copy()
        current_eef_rotmtx = self.data.site_xmat[self.eef_id].copy()
        current_eef_quat = np.empty(4)
        mujoco.mju_mat2Quat(current_eef_quat, current_eef_rotmtx)

        return current_eef_pos.copy(), current_eef_quat.copy()

    def _get_eef_achieved_velocity(self):
        vel = self.jac[:3, :] @ self.data.qvel[self.joint_ids]
        angular_vel = self.jac[3:, :] @ self.data.qvel[self.joint_ids]
        return vel.copy(), angular_vel.copy()

    def _compute_jacobian(self):
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))

        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.eef_id)
        jac = np.concatenate((jac_pos[:, self.joint_ids],
                              jac_rot[:, self.joint_ids]), axis=0)
        return jac

    def _compute_mass_matrix(self):
        """ Compute the mass matrix of the robot in operational space.
            M_x = (J * M^-1 * J^T)^-1

        Returns:
            M   (np.array): Mass matrix of the robot
            M_x (np.array): Inertia matrix in operational space
        """
        # M: Mass matrix of the robot
        nv = self.model.nv
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        M = M[self.joint_ids, :][:, self.joint_ids]

        # M_x: Inertia matrix in operational space
        M_inv = np.linalg.inv(M)
        Mx_inv = np.dot(self.jac, np.dot(M_inv, self.jac.T))

        # Inverse if matrix is non-singular (faster & more accurate) or
        # set singular values < (rcond * max(singular_values)) to 0
        # 1e-15 is the default value of rcond in numpy
        if abs(np.linalg.det(Mx_inv)) >= 1e-2:
            M_x = np.linalg.inv(Mx_inv)
        else:
            M_x = np.linalg.pinv(Mx_inv, rcond=1e-2)

        return M, M_x

    def _reach_joint_limit(
        self,
        target_pos: np.array,
        target_quat: np.array
    ):

        JOINT_LIMITS_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718,
                                     -2.8973, -0.0175, -2.8973]) + 0.1
        JOINT_LIMITS_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973,
                                     3.7525, 2.8973]) - 0.1

        delta_pos = target_pos - self.current_pos
        delta_ori = quaternion_error(target_quat, self.current_eef_quat)
        delta_wrench = np.concatenate([delta_pos, delta_ori])

        jac_inv = np.linalg.pinv(self.jac)

        delta_qpos = np.dot(jac_inv, delta_wrench)

        current_qpos = self.data.qpos[self.joint_ids]
        new_qpos = current_qpos + delta_qpos

        if np.any(new_qpos < JOINT_LIMITS_MIN) or \
           np.any(new_qpos > JOINT_LIMITS_MAX):
            if DEBUG_MODE:
                print('Desired_pose reaching joint limit:')
                print('min:\t', JOINT_LIMITS_MIN)
                print('new:\t', new_qpos)
                print('max:\t', JOINT_LIMITS_MAX)
            return True

        return False

    def _pd_control(
        self,
        x: np.array,
        dx: np.array,
        desired_x: np.array,
        desired_dx: np.array,
        gains: Dict,
        mode="position"
    ):
        if mode == "position":
            # F_r = kp * err_pos + kd * err_vel
            err_x = desired_x - x
            err_dx = desired_dx - dx

        elif mode == "orientation":
            # Tau_r = kp * err_ori + kd * err_angular_vel
            err_x = quaternion_error(desired_x, x)
            err_dx = desired_dx - dx

        elif mode == "nullspace":
            # F_null = kp * err_qpos + kd * err_qvel
            err_x = desired_x - x
            err_dx = desired_dx - dx

        else:
            raise ValueError("Invalid mode for pd control")

        return gains["kp"] * err_x + gains["kd"] * err_dx

    def compute_control_signal(
        self,
        target_pos: np.array,
        target_quat: np.array
    ):
        """ Return joint torques to achieve the target end-effector pose (in
            the world frame).
            Args:
                target_pos  (np.array): target end-effector position
                target_quat (np.array): target end-effector quaternion

            Returns:
                joint_torques (np.array): The joint torques to achieve the
                                          desired end-effector configuration.
        """

        self.jac = self._compute_jacobian()
        # Current EEF state
        self.current_pos, self.current_eef_quat = self._get_eef_achieved_pose()
        current_vel, current_angular_vel = self._get_eef_achieved_velocity()

        # if self._reach_joint_limit(target_pos, target_quat):
        #     return self.last_control_signal

        # Mass matrix and inertia matrix
        M_mass, M_x = self._compute_mass_matrix()

        # F_r = kp * err_pos + kd * err_vel
        desired_force = self._pd_control(
            x=self.current_pos, desired_x=target_pos,
            dx=current_vel, desired_dx=self.target_vel,
            gains=self.controller_gains["position"], mode="position")

        # Tau_r = kp * ori_err + kd * vel_err
        desired_torque = self._pd_control(
            x=self.current_eef_quat, desired_x=target_quat,
            dx=current_angular_vel, desired_dx=self.target_angular_vel,
            gains=self.controller_gains["orientation"], mode="orientation")

        desired_wrench = np.hstack([desired_force, desired_torque])

        # CHECKED: J^T * M_x * desired_wrench
        joint_torques = self.jac.T @ M_x @ desired_wrench

        # Cancel the effects of gravity
        joint_torques += self.data.qfrc_bias[self.joint_ids]

        #############################################
        # Null space: a secondary control signal
        #############################################
        current_qpos = self.data.qpos[self.joint_ids]
        target_qpos = self.nullspace_joint_config['qpos']
        current_qvel = self.data.qvel[self.joint_ids]
        target_qvel = self.nullspace_joint_config['vel']

        # Tau_null = kp * err_qpos + kd * err_qvel
        desired_torque_null = self._pd_control(
            x=current_qpos, desired_x=target_qpos,
            dx=current_qvel, desired_dx=target_qvel,
            gains=self.controller_gains["nullspace"], mode="nullspace")

        # CHECKED: Jac_T_pseudo_inv (pseudo-inverse) = M_x * J * M^-1
        Jac_T_pseudo_inv = M_x @ self.jac @ np.linalg.inv(M_mass)

        # CHECKED: (I - J^T * Jac_pseudo_inv) * desired_torque_null
        I_mat = np.eye(len(self.joint_ids))
        joint_torques += \
            (I_mat - self.jac.T @ Jac_T_pseudo_inv) @ desired_torque_null

        # Compute effective torque
        actuator_moment_inv = np.linalg.pinv(self.data.actuator_moment)
        actuator_moment_inv = \
            actuator_moment_inv[self.joint_ids, :][:, self.joint_ids]
        joint_torques = joint_torques @ actuator_moment_inv

        self.last_control_signal = joint_torques.copy()

        return joint_torques

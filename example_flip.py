
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import mujoco
import numpy as np
from os import path
from scipy.spatial.transform import Rotation as R

from controllers.os_controller import OSController


np.set_printoptions(precision=3)


def get_contact_force(model, data, sensor_name='force_ee'):
    """ Get contact force from relevant sensor

    Args:
        sensor_name (str): Sensor's name

    Returns:
        force (np.ndarray): Contact force
    """

    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR,
                                  sensor_name)

    sensor_dim = model.sensor_dim[sensor_id]
    # Sum all dims from the sensor before
    sensor_idx = np.sum(model.sensor_dim[:sensor_id])
    current_contact_force = np.array(data.sensordata[
        sensor_idx: sensor_idx + sensor_dim])
    return np.abs(current_contact_force)


def go_to(target_pos, target_quat, finger_width, control_steps,
          nstep=1, n_frame_skip=1):
    max_contact_force = 0
    for k in range(control_steps):
        delta_qpos = controller.compute_control_signal(
            target_pos, target_quat)

        data.ctrl[:7] = delta_qpos[:7]
        data.ctrl[7] = finger_width

        mujoco.mj_step(model, data, nstep=nstep)
        if k % n_frame_skip == 0:
            mujoco_renderer.render(render_mode)
        mujoco.mj_rnePostConstraint(model, data)

        max_contact_force = max(max_contact_force,
                                get_contact_force(model, data, 'force_ee')[0])

    mujoco_renderer.render(render_mode)
    return max_contact_force


if __name__ == '__main__':
    render_mode = "human"
    # render_mode = "rgb_array"
    default_camera_config = {
        "distance": 0.8,
        "azimuth": 180.0,
        "elevation": -15.0,
        "lookat": np.array([0.65, 0., 1.1])}

    assert_path = "assets/franka_emika_panda/scene.xml"
    assert_path = path.join(path.dirname(path.realpath(__file__)), assert_path)

    # Initialize simulation
    model = mujoco.MjModel.from_xml_path(assert_path)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)
    mujoco.mj_resetData(model, data)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    mujoco_renderer = MujocoRenderer(model, data, default_camera_config)
    mujoco.mj_step(model, data, nstep=10)
    mujoco_renderer.render(render_mode)

    #######################################################
    # NOTE: TASK: Testing object flipping
    #######################################################
    config = {
        "null": {"qpos": [0, -0.7853, 0, -2.35609, 0, 1.57079, 0.7853]},
        "gains": {"position": {"kp": 600., "kd": 48.99},
                  "orientation": {"kp": 600., "kd": 48.99},
                  "nullspace": {"kp": 1, "kd": 1}}}
    controller = OSController(model, data, config)

    # Modify the object position (freejoint)
    data.joint("cube").qpos[:3] = np.array([0.65, 0.138, 1.03])
    data.joint("cube").qpos[3:] = np.array([1, 0, 0, 0])
    mujoco.mj_step(model, data, nstep=1)
    mujoco_renderer.render(render_mode)

    ########################################
    # Move robot to starting pose
    ########################################
    current_eef_pos, _ = controller._get_eef_achieved_pose()
    initial_pos = np.array([0.66, 0.08, 1.0])
    initial_quat = R.from_euler(
        'xyz', [180, 0, 90], degrees=True).as_quat()[[3, 0, 1, 2]]

    # 0.02 is helf length of gripper
    target_pos = initial_pos.copy()
    target_quat = initial_quat.copy()
    go_to(target_pos, target_quat, finger_width=0.001, control_steps=500,
          nstep=1, n_frame_skip=5)

    trajectory_push = np.array([[0, 0.027, 0], [0, 0.015, 0], [0, 0.005, 0]])
    trajectory_flip = np.array([
        [0., 0.012, 0.0], [0., 0.005, 0.025], [0., 0.025, 0.01],
        [0., 0.03, 0.01]])

    ########################################
    # Pushing
    ########################################
    max_contact_force = 0
    current_eef_pos, current_eef_quat = controller._get_eef_achieved_pose()
    target_pos = current_eef_pos
    for action in trajectory_push:
        target_pos = target_pos + action
        target_quat = initial_quat.copy()

        max_contact_force = \
            go_to(target_pos, target_quat, finger_width=0.001,
                  control_steps=200, nstep=1, n_frame_skip=5)

        print(f"Max contact force  = {max_contact_force}")
        current_eef_pos, current_eef_quat = controller._get_eef_achieved_pose()
        error = np.linalg.norm(current_eef_pos - target_pos)
        print(f"current_eef_pos: {current_eef_pos} | target_pos: {target_pos} | error: {error}")

    ########################################
    # Flipping
    ########################################
    max_contact_force = 0
    current_eef_pos, current_eef_quat = controller._get_eef_achieved_pose()
    target_pos = current_eef_pos
    for action in trajectory_flip:
        # Not the current action, but the target position
        target_pos = target_pos + action
        target_quat = initial_quat.copy()

        max_contact_force = \
            go_to(target_pos, target_quat, finger_width=0.001,
                  control_steps=100, nstep=1, n_frame_skip=5)

        print(f"Max contact force  = {max_contact_force}")
        current_eef_pos, current_eef_quat = controller._get_eef_achieved_pose()
        error = np.linalg.norm(current_eef_pos - target_pos)
        print(f"current_eef_pos: {current_eef_pos} | target_pos: {target_pos} | error: {error}")

    ########################################
    # Free moving the view
    ########################################
    for _ in range(5000):
        go_to(target_pos, target_quat, finger_width=0.001, control_steps=100,
              nstep=1, n_frame_skip=5)

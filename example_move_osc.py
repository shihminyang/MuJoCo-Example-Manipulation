from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import mujoco
import numpy as np
from os import path
from scipy.spatial.transform import Rotation as R

from controllers.os_controller import OSController


def go_to(target_pos, target_quat, finger_width, control_steps,
          nstep=1, n_frame_skip=1):

    for k in range(control_steps):
        delta_qpos = controller.compute_control_signal(
            target_pos, target_quat)

        data.ctrl[:7] = delta_qpos[:7]
        data.ctrl[7] = finger_width

        mujoco.mj_step(model, data, nstep=nstep)
        if k % n_frame_skip == 0:
            mujoco_renderer.render(render_mode)
        mujoco.mj_rnePostConstraint(model, data)

    mujoco_renderer.render(render_mode)


if __name__ == '__main__':
    render_mode = "human"
    # render_mode = "rgb_array"
    default_camera_config = {
        "distance": 1.2,
        "azimuth": 90.0,
        "elevation": -15.0,
        "lookat": np.array([0.55, 0., 1.5])}

    assert_path = "assets/franka_emika_panda/scene.xml"
    assert_path = path.join(path.dirname(path.realpath(__file__)), assert_path)

    # Initialize simulation
    model = mujoco.MjModel.from_xml_path(assert_path)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)
    mujoco.mj_resetData(model, data)
    mujoco.mj_resetDataKeyframe(model, data, 1)

    mujoco_renderer = MujocoRenderer(model, data, default_camera_config)
    mujoco.mj_step(model, data, nstep=10)
    mujoco_renderer.render(render_mode)

    #######################################################
    # NOTE: TASK: Test the operational space controller
    # random_move / forward_back / up_down / left_right
    #######################################################
    i = 0
    test_mode = ["random_move", "forward_back", "up_down", "left_right"][i]

    config = {
        "null": {"qpos": [0, -0.7853, 0, -2.35609, 0, 1.57079, 0.7853]},
        "gains": {"position": {"kp": 600., "kd": 48.99},
                  "orientation": {"kp": 600., "kd": 48.99},
                  "nullspace": {"kp": 1, "kd": 1}}}
    controller = OSController(model, data, config)

    current_eef_pos, _ = controller._get_eef_achieved_pose()
    position = current_eef_pos.copy()

    ratio = -0.05
    for _ in range(30):
        if test_mode == 'random_move':
            position_x = np.random.uniform(0.35, 0.60)
            position_y = np.random.uniform(-0.3, 0.3)
            position_z = np.random.uniform(1.2, 1.50)
            position = np.array([position_x, position_y, position_z])

        elif test_mode == 'forward_back':
            ratio *= -1
            position[0] += ratio

        elif test_mode == 'left_right':
            ratio *= -1
            position[1] += ratio

        elif test_mode == 'up_down':
            ratio *= -1
            position[2] += ratio

        quat = np.zeros(4,)
        mat = obj_rot = R.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix().flatten()
        mujoco.mju_mat2Quat(quat, mat)

        target_pos = position.copy()
        target_quat = quat.copy()

        go_to(target_pos, target_quat, finger_width=0.04, control_steps=600,
              nstep=1, n_frame_skip=2)

        current_eef_pos, _ = controller._get_eef_achieved_pose()
        error = np.linalg.norm(current_eef_pos - target_pos)
        print(f"Target position: {position} | Current position: {current_eef_pos} | Error: {error}")

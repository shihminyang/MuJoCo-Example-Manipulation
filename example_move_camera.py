from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import mujoco
import numpy as np
from os import path
from scipy.spatial.transform import Rotation as R


def rand_vector(dim, _range):
    return np.random.rand(dim) * (_range[1] - _range[0]) + _range[0]


if __name__ == '__main__':
    render_mode = "human"
    default_camera_config = {
        "distance": 1.2,
        "azimuth": 90.0,
        "elevation": -15.0,
        "lookat": np.array([0.55, 0., 1.5])}

    model_path = path.join(path.dirname(path.realpath(__file__)),
                           "assets/franka_emika_panda/scene.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    mujoco_renderer = MujocoRenderer(model, data, default_camera_config)

    camera_name = "sensor"
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA,
                                  camera_name)
    camera_init_pos = model.camera(camera_id).pos.copy()
    camera_init_quat = model.camera(camera_id).quat.copy()
    camera_init_euler = R.from_quat(camera_init_quat[[3, 0, 1, 2]]
                                    ).as_euler("XYZ")
    camera_init_mat = R.from_quat(camera_init_quat[[3, 0, 1, 2]]).as_matrix()

    i = 0
    while True:
        i += 1
        mujoco.mj_step(model, data, nstep=2)
        mujoco_renderer.render(render_mode)

        if i % 10 != 0:
            continue

        input("Press Enter to move the camera")

        xyz_range = np.array([[-0.05, -0.05, -0.1], [0.05, 0.05, 0.1]])
        delta_pos = rand_vector(3, xyz_range)

        xyz_range = np.array([[-5., -5., -5], [5., 5, 5]]) * np.pi / 180
        delta_euler = rand_vector(3, xyz_range)
        delta_mat = R.from_euler("XYZ", delta_euler).as_matrix()

        pos = camera_init_pos + delta_pos
        # (qx, qy, qz, qw) -> (qw, qx, qy, qz)
        quat = R.from_matrix(np.dot(camera_init_mat, delta_mat)
                             ).as_quat()[[1, 2, 3, 0]]

        model.camera(camera_id).pos = pos
        model.camera(camera_id).quat = quat

        mujoco.mj_step(model, data, nstep=1)
        mujoco_renderer.render(render_mode)

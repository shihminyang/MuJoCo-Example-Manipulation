import cv2
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import mujoco
import numpy as np
from os import path
from scipy.spatial.transform import Rotation as R


def rand_vector(dim, _range):
    return np.random.rand(dim) * (_range[1] - _range[0]) + _range[0]


class Camera:
    def __init__(self, model, mujoco_renderer, camera_name) -> None:
        self.model = model
        self.mujoco_renderer = mujoco_renderer
        self.camera_name = camera_name
        self.camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.init_pos = model.camera(self.camera_id).pos.copy()
        self.init_quat = model.camera(self.camera_id).quat.copy()
        self.init_euler = R.from_quat(self.init_quat[[3, 0, 1, 2]]
                                      ).as_euler("XYZ")
        self.init_mat = R.from_quat(self.init_quat[[3, 0, 1, 2]]).as_matrix()

        self.workspace = {'depth_min': 0, 'depth_max': 3}

    def _switch_camera(self, camera_id):
        self.mujoco_renderer.camera_id = camera_id

    def _convert_to_real_depth(self, depth_map):
        """ (see https://github.com/google-deepmind/dm_control/blob/main/dm_control/mujoco/engine.py#L893)
            Convert depth_map (MuJoCo default depth_array) into real distances

        Args:
            depth_map   (np.array): depth map [0, 1] (returned by MuJoCo
                                    render)
        Return:
            depth_map   (np.array): depth map with real distances
        """

        # Get the distances to the near and far clipping planes
        extent = self.model.stat.extent
        far = self.model.vis.map.zfar * extent
        near = self.model.vis.map.znear * extent
        return near / (1. - depth_map * (1. - near / far))

    def get_color_image(self):
        self._switch_camera(self.camera_id)
        image = self.mujoco_renderer.render('rgb_array')

        # Normalize to [0, 1]
        image = image.astype(float) / 255.
        print(image.shape)
        self._switch_camera(-1)
        return image

    def get_depth_image(self):
        self._switch_camera(self.camera_id)
        image = self.mujoco_renderer.render('depth_array')
        image = self._convert_to_real_depth(image)

        # Limit in the workspace
        image[image < self.workspace['depth_min']] = 0
        image[image > self.workspace['depth_max']] = 0
        self._switch_camera(-1)
        return image

    def set_pose(self, pos, quat):
        self.model.camera(self.camera_id).pos = pos
        self.model.camera(self.camera_id).quat = quat

    def show_image(self,
                   color_image: np.array = None,
                   depth_image: np.array = None, sec=0):
        if color_image is not None:
            color_image = (color_image * 255).astype(np.uint8)
            cv2.imshow("Color Image",
                       cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
        if depth_image is not None:
            cv2.imshow(
                "Depth Image",
                (depth_image / np.max(depth_image) * 255).astype(np.uint8))

        cv2.waitKey(sec)


if __name__ == '__main__':
    render_mode = "human"
    # render_mode = "rgb_array"
    default_camera_config = {
        "distance": 1.2,
        "azimuth": 90.0,
        "elevation": -15.0,
        "lookat": np.array([0.65, 0., 1.05])}

    image_width = 1280
    image_height = 720

    model_path = path.join(path.dirname(path.realpath(__file__)),
                           "assets/franka_emika_panda/scene.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    model.vis.global_.offwidth = image_width
    model.vis.global_.offheight = image_height
    mujoco_renderer = MujocoRenderer(model, data, default_camera_config,
                                     image_width, image_height)

    camera_1 = Camera(model, mujoco_renderer, "sensor_1")
    camera_2 = Camera(model, mujoco_renderer, "sensor_2")
    camera_3 = Camera(model, mujoco_renderer, "sensor_3")
    cameras = [camera_1, camera_2, camera_3]

    i = 0
    while True:
        i += 1
        mujoco.mj_step(model, data, nstep=2)
        mujoco_renderer.render(render_mode)

        if i % 10 != 0:
            continue

        # Random select camera
        k = np.random.randint(len(cameras), size=1)[0]
        camera = cameras[k]
        print("Selected camera: ", camera.camera_name)

        # Random delta pose
        xyz_range = np.array([[-0.05, -0.05, -0.1], [0.05, 0.05, 0.1]])
        delta_pos = rand_vector(3, xyz_range)

        xyz_range = np.array([[-5., -5., -5], [5., 5, 5]]) * np.pi / 180
        delta_euler = rand_vector(3, xyz_range)
        delta_mat = R.from_euler("XYZ", delta_euler).as_matrix()

        # Set new pose
        pos = camera.init_pos + delta_pos
        # (qx, qy, qz, qw) -> (qw, qx, qy, qz)
        quat = R.from_matrix(np.dot(camera.init_mat, delta_mat)
                             ).as_quat()[[1, 2, 3, 0]]
        camera.set_pose(pos, quat)

        # Update simulation
        mujoco.mj_step(model, data, nstep=1)
        mujoco_renderer.render(render_mode)

        # Take a picture
        color_image = camera.get_color_image()
        depth_image = camera.get_depth_image()
        camera.show_image(color_image, depth_image)

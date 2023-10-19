
import os 
import time
import math
import numpy as np

from isaacgym import gymapi
from isaacgym import gymtorch

import torch 
from sim.base_sim import BaseSimulator


class IKSimulator(BaseSimulator):
    def __init__(self, headless=False, num_envs=64, cam_params=None):
        super(IKSimulator, self).__init__()
        self.gym = gymapi.acquire_gym()

        sim_params = gymapi.SimParams()

        # set common parameters
        sim_params.dt = 1 / 60
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.use_gpu_pipeline = True

        # set PhysX-specific parameters
        sim_params.physx.use_gpu = True
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0

        self.sim = self.gym.create_sim(0, -1 if headless and cam_params['render_mode'] is None else 0, 
                                       gymapi.SIM_PHYSX, sim_params)
        # self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

        # configure the ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
        plane_params.distance = 0
        plane_params.static_friction = 1
        plane_params.dynamic_friction = 1
        plane_params.restitution = 0

        # create the ground plane
        self.gym.add_ground(self.sim, plane_params)

        asset_root = "/home/orr/research/isaacgym/assets"

        # Panda asset
        asset_file = "urdf/franka_description/robots/franka_panda.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        self.franka_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # configure franka dofs
        franka_dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
        franka_lower_limits = franka_dof_props["lower"]
        franka_upper_limits = franka_dof_props["upper"]
        franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

        # default dof states and position targets
        franka_num_dofs = self.gym.get_asset_dof_count(self.franka_asset)
        default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
        default_dof_pos[:7] = franka_mids[:7]
        # grippers open
        default_dof_pos[7:] = franka_upper_limits[7:]

        self.default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
        self.default_dof_state["pos"] = default_dof_pos

        # Box asset (needs to be created with set_gt)
        self.box_asset = None
        self.box_pose = None

        # Env setup
        self.num_envs = num_envs
        self.envs_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        self.env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        self.env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # Robot pose
        self.franka_pose = gymapi.Transform()
        self.franka_pose.p = gymapi.Vec3(-0.6, 0, 0)

        self.envs = []
        self.robots = []
        self.hand_idxs = []
        self.boxes = []
        self.cameras = []
        self.max_cameras = 50  # Arbitrary number to limit rendering time

        if cam_params['render_mode'] is not None:
            # Single Env Camera Properties
            self.env_cam_props = gymapi.CameraProperties()
            self.env_cam_props.width = 512 if 'high_q' in cam_params['render_mode'] else 128
            self.env_cam_props.height = 512 if 'high_q' in cam_params['render_mode'] else 128

            # camera position and target
            self.env_cam_pos = gymapi.Vec3(*cam_params['pos'])
            self.env_cam_target = gymapi.Vec3(*cam_params['target'])
        else:
            self.env_cam_props = None

        # EE Target pose
        self.target_ee = None
        self.ball_asset = None
        self.ball_pose = None

        # Viewer
        if not headless:
            cam_props = gymapi.CameraProperties()
            self.viewer = self.gym.create_viewer(self.sim, cam_props)
        else:
            self.viewer = None

        self.dof_states = None
        self.rb_states = None

    def set_gt(self, gt):
        # gt = [box shape for single-box|list of box shapes for multi-box|urdf path, 
        #       box pose for single box|list of box poses for multi-box, 
        #       target_ee (which is also offset for object), 
        #       rotation]

        box_asset_options = gymapi.AssetOptions()
        box_asset_options.disable_gravity = True
        box_asset_options.fix_base_link = True

        if isinstance(gt[0], list):
            if isinstance(gt[0][0], (list, tuple)):
                # Build multi-box object
                self.box_asset = []
                for x, y, z in gt[0]:
                    self.box_asset.append(self.gym.create_box(self.sim, x, y, z, box_asset_options))
            else:
                # Build single-box object
                box_x, box_y, box_z = gt[0]
                self.box_asset = self.gym.create_box(self.sim, box_x, box_y, box_z, box_asset_options)
        
        elif isinstance(gt[0], str):
            # Build urdf-based object
            box_asset_root, box_asset_file = os.path.split(gt[0])
            self.box_asset = self.gym.load_asset(self.sim, box_asset_root, box_asset_file, box_asset_options)
        else:
            raise ValueError(f"Unknown gt objcet type: {type(gt[0])}")

        # Box pose
        if isinstance(gt[1][1], (list, tuple)):
            # multi-box object
            self.box_pose = []
            for x, y, z in gt[1]:
                box_pose = gymapi.Transform()
                box_pose.p = gymapi.Vec3(x, y, z)
                if len(gt) > 3:
                    box_pose.r = gymapi.Quat(*gt[3])
                    box_pose.p = box_pose.r.rotate(box_pose.p)
                box_pose.p += gymapi.Vec3(*gt[2])
                self.box_pose.append(box_pose)
        else:
            # Single box object
            self.box_pose = gymapi.Transform()
            self.box_pose.p = gymapi.Vec3(*gt[1])
            if len(gt) > 3:
                box_pose.r = gymapi.Quat(*gt[3])
                box_pose.p = box_pose.r.rotate(box_pose.p)

        # Target pose (used to calculate score)
        self.target_ee = torch.tensor(gt[2], dtype=torch.float, device=torch.device('cuda:0')) # TODO: correct device assignment
        self.ball_asset = self.gym.create_sphere(self.sim, 0.01, box_asset_options)
        self.ball_pose = gymapi.Transform()
        self.ball_pose.p = gymapi.Vec3(*gt[2])

    def _create_envs(self):
        assert not len(self.envs), "Envs can only be created once!"
        assert self.target_ee is not None, "Must set_gt before creating envs!"

        # Create envs
        for ind in range(self.num_envs):
            env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, self.envs_per_row)
            self.envs.append(env)

            # add franka
            robot = self.gym.create_actor(env, self.franka_asset, self.franka_pose, "robot", ind, 2)
            self.robots.append(robot)
            self.gym.set_actor_dof_states(env, robot, self.default_dof_state, gymapi.STATE_ALL)
            
            # Get global index of hand in rigid body state tensor
            hand_idx = self.gym.find_actor_rigid_body_index(env, robot, "panda_grasptarget", gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)

            if self.box_asset is not None and self.box_pose is not None:
                # color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
                color = gymapi.Vec3(0, 0, 1)
                if isinstance(self.box_asset, list):
                    assert isinstance(self.box_pose, list), "Box asset and pose must be both lists or both not lists"
                    # multi-box object
                    for ii, (box_asset, box_pose) in enumerate(zip(self.box_asset, self.box_pose)):
                        self.boxes.append(self.gym.create_actor(env, box_asset, box_pose, f"w_side_{ii}", ind, 0))
                        self.gym.set_rigid_body_color(env, self.boxes[-1], 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                else:
                    # single-box object
                    self.boxes.append(self.gym.create_actor(env, self.box_asset, self.box_pose, "box", ind, 0))
                    self.gym.set_rigid_body_color(env, self.boxes[-1], 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            
            # Add ball to mark target EE pose. Set collision group so it doesn't collide with the robot
            ball_handle = self.gym.create_actor(env, self.ball_asset, self.ball_pose, "ball", ind + self.num_envs, 0)
            self.gym.set_rigid_body_color(env, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 1, 0))

            # add camera
            if self.env_cam_props is not None and ind < self.max_cameras:
                # Note: can't handle too many cameras when rendering later, limiting arbitrarily to 50 cameras
                cam_handle = self.gym.create_camera_sensor(env, self.env_cam_props)
                self.gym.set_camera_location(cam_handle, env, self.env_cam_pos, self.env_cam_target)
                self.cameras.append(cam_handle)

        self.gym.prepare_sim(self.sim)

        # DoF state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)

        # Rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

    def render(self):
        assert self.viewer is not None, "Cannot render when running headless"
        self.gym.draw_viewer(self.viewer, self.sim, True)

    def render_dist(self, samples):        
        res, t = self.test_objects(samples, timer=True)
        
        # Only show the best samples
        if len(samples) > self.max_cameras:
            res, t = self.test_objects(samples[res.topk(self.max_cameras).indices], timer=True)

        if self.env_cam_props is None:
            return None, res, t
        
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # Get camera image
        # cams = np.random.choice(self.envs, num_samples)
        imgs = [self.gym.get_camera_image(self.sim, self.envs[cam_ind], 0, gymapi.IMAGE_COLOR) for cam_ind in range(len(samples))]
        
        return imgs, res, t

    def test_objects(self, object_batch, action=None, timer=False):
        """
        :param object_batch: Batch of objects to manipulate (in this case: robot configs)
        :param action: unused 
        :return: Probability of manipulation success (or score) per object
        """
        if not len(self.envs):
            self._create_envs()
        
        if timer:
            st = time.time() 
        # Needs to be (num_envs x 9, 2). 9 = 7-DoF Panda + 2 dims for gripper joints, 2 = pos, vel
        # Also needs to be torch.float32

        bsz = object_batch.shape[0]
        # Add zeros for finger joints
        object_batch = torch.cat((object_batch, torch.zeros_like(object_batch[:, -2:])), dim=-1)
        # Add zeros for joint velocities
        object_batch = torch.stack((object_batch.view(-1), torch.zeros_like(object_batch.view(-1))), dim=-1)
        self.dof_states[:object_batch.shape[0], :] = object_batch
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))
        # self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(object_batch))  

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # To calculate score, check if pose is close to what we set it to be (otherwise robot has collided and moved)
        # But also, need to check that it is close to the goal ee pos!
        
        ee_pos_cur = self.rb_states[self.hand_idxs, :3]

        legal = torch.isclose(self.dof_states[:object_batch.shape[0]], object_batch).view(bsz, 9, 2)[:, :7, 0]
        ee_dist = torch.exp(-torch.norm(self.target_ee - ee_pos_cur, dim=-1))
        # print(f'Minimal EE dist: {(self.target_ee - ee_pos_cur).norm(dim=-1).min()}')
        # print(f'EE pos: {ee_pos_cur[(self.target_ee - ee_pos_cur).norm(dim=-1).argmin()]}')

        score = torch.all(legal, dim=1).float() * ee_dist[:bsz]

        # update the viewer
        if self.viewer is not None:         
            while not self.gym.query_viewer_has_closed(self.viewer):
                self.gym.step_graphics(self.sim)
                self.render()
                time.sleep(0.1)
            a = 1
        
        if timer:
            return score, time.time() - st
        else:
            return score
        
    def test_and_show(self, obj=None, action=None, title=None):
        pass

    def close(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def set_camera_to_env(self, env_id):
        # TODO: This doesn't work - probably can't move viewer camera this way
        assert self.viewer is not None
        cam = self.gym.get_viewer_camera_handle(self.viewer)
        env = self.envs[env_id]
        self.gym.set_camera_location(cam, env, gymapi.Vec3(1., 1., 1.), gymapi.Vec3(-0.6, 0., 0.))


if __name__ == '__main__':
    sim = IKSimulator()
    obj_batch = torch.zeros((576, 2), dtype=torch.float32, device=torch.device('cuda:0'))
    sim.test_objects(obj_batch)
    sim.render()
    pass
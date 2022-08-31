from copy import deepcopy
import cv2
import carla

import random
import torch
import numpy as np
import pygame
import json
import math

# from utils import lts_rendering
# from utils.map_utils import MapImage, encode_npy_to_pil, PIXELS_PER_METER
# from autopilot import AutoPilot
from data_agent import DataAgent
# from cosim_wrapper import CosimAgent
from srunner.scenariomanager.timer import GameTime

def get_entry_point():
    return 'AccelAgent'


class AccelAgent(DataAgent):

    def setup(self, path_to_conf_file, route_index=None):
        super().setup(path_to_conf_file, route_index)
        self.synchronization = None

    def sensors(self):
        return [{
                    'type': 'sensor.opendrive_map',
                    'reading_frequency': 1e-6,
                    'id': 'hd_map'
                    },
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'imu'
                    },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                    },
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                    },

                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z':2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                    'id': 'rgb_front'
                },
                # {
                #     'type': 'sensor.camera.rgb',
                #     'x': 1.3, 'y': 0.0, 'z':2.3,
                #     'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                #     'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                #     'id': 'rgb_left'
                # },
                # {
                #     'type': 'sensor.camera.rgb',
                #     'x': 1.3, 'y': 0.0, 'z':2.3,
                #     'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                #     'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                #     'id': 'rgb_right'
                # },
                {
                    'type': 'sensor.lidar.ray_cast',
                    'x': 1.3, 'y': 0.0, 'z': 2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                    'rotation_frequency': 20,
                    'points_per_second': 1200000,
                    'id': 'lidar'
                },
                # {
                #     'type': 'sensor.camera.semantic_segmentation',
                #     'x': 1.3, 'y': 0.0, 'z':2.3,
                #     'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                #     'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                #     'id': 'semantics_front'
                # },
                # {
                #     'type': 'sensor.camera.semantic_segmentation',
                #     'x': 1.3, 'y': 0.0, 'z':2.3,
                #     'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                #     'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                #     'id': 'semantics_left'
                # },
                # {
                #     'type': 'sensor.camera.semantic_segmentation',
                #     'x': 1.3, 'y': 0.0, 'z':2.3,
                #     'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                #     'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                #     'id': 'semantics_right'
                # },
                # {
                #     'type': 'sensor.camera.depth',
                #     'x': 1.3, 'y': 0.0, 'z':2.3,
                #     'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                #     'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                #     'id': 'depth_front'
                # },
                # {
                #     'type': 'sensor.camera.depth',
                #     'x': 1.3, 'y': 0.0, 'z':2.3,
                #     'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                #     'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                #     'id': 'depth_left'
                # },
                # {
                #     'type': 'sensor.camera.depth',
                #     'x': 1.3, 'y': 0.0, 'z':2.3,
                #     'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                #     'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                #     'id': 'depth_right'
                # },
        ]

    def forward_step(self, action):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        input_data = self.sensor_interface.get_data()

        timestamp = GameTime.get_time()

        if not self.wallclock_t0:
            self.wallclock_t0 = GameTime.get_wallclocktime()
        wallclock = GameTime.get_wallclocktime()
        wallclock_diff = (wallclock - self.wallclock_t0).total_seconds()

        # print('======[Agent] Wallclock_time = {} / {} / Sim_time = {} / {}x'.format(wallclock, wallclock_diff, timestamp, timestamp/(wallclock_diff+0.001)))

        result = self.run_step(input_data, action, timestamp)
        # print("Topdown shape: ", topdown.shape)
        if result: 
            control, rgb, state, player_ind = result
            control.manual_gear_shift = False

        return result

    def tick(self, input_data):
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]
        if (math.isnan(compass) == True): # simulation bug
            compass = 0.0

        result = {
                'gps': gps,
                'speed': speed,
                'compass': compass,
                }

        # if self.save_path is not None:
        #     rgb = []
        #     semantics = []
        #     depth = []
        #     for pos in ['left', 'front', 'right']:
        #         rgb_cam = 'rgb_' + pos
        #         semantics_cam = 'semantics_' + pos
        #         depth_cam = 'depth_' + pos
        #         semantics_img = input_data[semantics_cam][1][:, :, 2]
        #         depth_img = input_data[depth_cam][1][:, :, :3] 
        #         _semantics = np.copy(semantics_img)
        #         _depth = self._get_depth(depth_img)
        #         self._change_seg_tl(_semantics, _depth, self._active_traffic_light)

        #         rgb.append(cv2.cvtColor(input_data[rgb_cam][1][:, :, :3], cv2.COLOR_BGR2RGB))
        #         semantics.append(_semantics)
        #         depth.append(depth_img)

        #     rgb = np.concatenate(rgb, axis=1)
        #     semantics = np.concatenate(semantics, axis=1)
        #     depth =  np.concatenate(depth, axis=1)

        result['topdown'] = self.render_BEV()
        lidar = input_data['lidar']
        cars = self.get_bev_cars(lidar=lidar)

        # result.update({'lidar': lidar,
        #                 'rgb': rgb,
        #                 'cars': cars,
        #                 'semantics': semantics,
        #                 'depth': depth})

        return result

    @torch.no_grad()
    def run_step(self, input_data, action, timestamp):
        self.step += 1
        if not self.initialized and ('hd_map' in input_data.keys()):
            self._init(input_data['hd_map'])
            # else:
            #     control = carla.VehicleControl()
            #     control.steer = 0.0
            #     control.throttle = 0.0
            #     control.brake = 1.0
            #     return control

        control = self._get_control(action, input_data)
        tick_data_tmp = self.tick(input_data)
        self.update_gps_buffer(control, tick_data_tmp['compass'], tick_data_tmp['speed'])

        if self.synchronization.sumo.player_has_result():
            state, player_ind = self.synchronization.get_state()
            tick_data = self.tick(input_data)

            if self.step % 10 == 0:
                self.shuffle_weather()
            
        return control, tick_data['topdown'], state, player_ind

    def _get_control(self, action, input_data, steer=None, throttle=None,
                        vehicle_hazard=None, light_hazard=None, walker_hazard=None, stop_sign_hazard=None):
        # insert missing controls
        if vehicle_hazard is None or light_hazard is None or walker_hazard is None or stop_sign_hazard is None:
            brake = self._get_brake(vehicle_hazard, light_hazard, walker_hazard, stop_sign_hazard) # privileged
        else:
            brake = vehicle_hazard or light_hazard or walker_hazard or stop_sign_hazard

        ego_vehicle_waypoint = self.world_map.get_waypoint(self._vehicle.get_location())
        self.junction = ego_vehicle_waypoint.is_junction

        speed = input_data['speed'][1]['speed']
        # target_speed = self.target_speed_slow if self.junction else self.target_speed_fast
        target_speed = action + speed if not brake else 0

        pos = self._get_position(input_data['gps'][1][:2])
        self.gps_buffer.append(pos)
        pos = np.average(self.gps_buffer, axis=0) # Denoised position

        self._waypoint_planner.load()
        waypoint_route = self._waypoint_planner.run_step(pos)
        near_node, near_command = waypoint_route[1] if len(waypoint_route) > 1 else waypoint_route[0] # needs HD map
        self._waypoint_planner.save()
        
        self._waypoint_planner_extrapolation.load()
        self.waypoint_route_extrapolation = self._waypoint_planner_extrapolation.run_step(pos)
        self._waypoint_planner_extrapolation.save()

        if throttle is None:
            throttle = self._get_throttle(brake, target_speed, speed)

            # hack for steep slopes
            if (self._vehicle.get_transform().rotation.pitch > self.slope_pitch):
                throttle += self.slope_throttle

        if steer is None:
            theta = input_data['imu'][1][-1]
            if math.isnan(theta):  # simulation bug
                theta = 0.0
            steer = self._get_steer(brake, waypoint_route, pos, theta, speed)
            steer_extrapolation = self._get_steer_extrapolation(waypoint_route, pos, theta, speed)

        self.steer_buffer.append(steer)

        control = carla.VehicleControl()
        control.steer = np.mean(self.steer_buffer) + self.steer_noise * np.random.randn()
        control.throttle = throttle
        control.brake = float(brake)

        self.steer = control.steer
        self.throttle = control.throttle
        self.brake = control.brake
        self.target_speed = target_speed

        self._save_waypoints()

        if((self.step % self.save_freq == 0) and (self.save_path is not None)):
            command_route = self._command_planner.run_step(pos)
            # far command is not always accurate
            far_node, far_command  = command_route[1] if len(command_route) > 1 else command_route[0]
            if (far_node != self.far_node_prev).all():
                self.far_node_prev = far_node
                self.commands.append(far_command.value)

            if self.render_bev==False:
                tick_data = self.tick(input_data)
            else:
                tick_data = self.tick(input_data, self.future_states)
            self.save(far_node, steer, throttle, brake, target_speed, tick_data)

        return control

    # def tick(self, input_data):
    #     result = super().tick(input_data)

    #     if self.save_path is not None:
    #         rgb = []
    #         semantics = []
    #         depth = []
    #         for pos in ['left', 'front', 'right']:
    #             rgb_cam = 'rgb_' + pos
    #             semantics_cam = 'semantics_' + pos
    #             depth_cam = 'depth_' + pos
    #             semantics_img = input_data[semantics_cam][1][:, :, 2]
    #             depth_img = input_data[depth_cam][1][:, :, :3] 
    #             _semantics = np.copy(semantics_img)
    #             _depth = self._get_depth(depth_img)
    #             self._change_seg_tl(_semantics, _depth, self._active_traffic_light)

    #             rgb.append(cv2.cvtColor(input_data[rgb_cam][1][:, :, :3], cv2.COLOR_BGR2RGB))
    #             semantics.append(_semantics)
    #             depth.append(depth_img)

    #         rgb = np.concatenate(rgb, axis=1)
    #         semantics = np.concatenate(semantics, axis=1)
    #         depth =  np.concatenate(depth, axis=1)

    #         result['topdown'] = self.render_BEV()
    #         lidar = input_data['lidar']
    #         cars = self.get_bev_cars(lidar=lidar)

    #         result.update({'lidar': lidar,
    #                         'rgb': rgb,
    #                         'cars': cars,
    #                         'semantics': semantics,
    #                         'depth': depth})

    #     return result


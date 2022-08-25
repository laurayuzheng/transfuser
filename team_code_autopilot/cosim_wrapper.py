from copy import deepcopy
import cv2
import carla

import random
import torch
import numpy as np
import pygame
import json

from utils import lts_rendering
from utils.map_utils import MapImage, encode_npy_to_pil, PIXELS_PER_METER
from autopilot import AutoPilot

from team_code_autopilot.data_agent import DataAgent


def get_entry_point():
    return 'CosimAgent'


class CosimAgent(DataAgent):
    def setup(self, path_to_conf_file, route_index=None):
        super().setup(path_to_conf_file, route_index)
        self.synchronization = None

        if self.save_path is not None:
            (self.save_path / 'traffic_info').mkdir()

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not ('hd_map' in input_data.keys()) and not self.initialized:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            return control

        control = super().run_step(input_data, timestamp)

        if self.step % self.save_freq == 0 and self.synchronization.sumo.player_has_result():
            if self.save_path is not None:
                tick_data = self.tick(input_data)
                self.save_sensors(tick_data)
                self.shuffle_weather()
            
        return control

    def save_sensors(self, tick_data):
        super().save_sensors(tick_data)
        frame = self.step // self.save_freq
        state, player_ind = self.synchronization.get_state()
        traffic_info = {
            'player_lane_state': state, 
            'player_ind_in_lane': player_ind, 
            'fuel_consumption': self.synchronization.sumo.get_playerlane_fuel_consumption()
        }
        self.save_labels(self.save_path / 'traffic_info' / ('%04d.json' % frame), traffic_info)
        



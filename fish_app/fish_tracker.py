import time
from collections import deque
import numpy as np

class FishTracker:
    def __init__(self, track_id: int, view_type: str, config):
        self.id = track_id
        self.view_type = view_type
        self.config = config
        self.positions = deque(maxlen=int(config['video']['fps'] * config['tracker']['memory']))
        self.last_seen = time.time()
        self.state = 'HEALTHY'
        self.detected_class = None
        self.time_in_top_zone = 0.0
        self.time_in_bottom_zone = 0.0
        self.time_sinking_with_movement = 0.0

    def update(self, box, detected_class=None):
        self.last_seen = time.time()
        # box is [x, y, w, h] or [x1, y1, x2, y2]?
        # DeepStream NvDsObjectMeta rect_params gives top, left, width, height
        # We need center x, y
        cx = box[0] + box[2] / 2
        cy = box[1] + box[3] / 2
        self.positions.append((cx, cy))
        if detected_class is not None:
            self.detected_class = detected_class

    def check_behavior(self):
        if not self.positions:
            return

        current_y = self.positions[-1][1]
        frame_h = self.config['video']['height']
        
        behavior_cfg = self.config['behavior']
        dead_cfg = behavior_cfg['dead']
        sick_cfg = behavior_cfg['sick']

        DEAD_VELOCITY_THRESHOLD = float(dead_cfg['velocity_threshold'])
        DEAD_TIME_THRESHOLD = float(dead_cfg['time_threshold'])
        TOP_ZONE_THRESHOLD = float(sick_cfg['top_zone'])
        BOTTOM_ZONE_THRESHOLD = float(sick_cfg['bottom_zone'])
        FRAME_RATE = self.config['video']['fps']

        if current_y > frame_h * BOTTOM_ZONE_THRESHOLD:
            self.time_sinking_with_movement += 1.0 / FRAME_RATE
            if len(self.positions) > 5:
                recent = np.array(list(self.positions)[-int(FRAME_RATE*3):])
                std_x = np.std(recent[:, 0])
                std_y = np.std(recent[:, 1])
                
                moving = (std_x >= DEAD_VELOCITY_THRESHOLD or 
                          std_y >= DEAD_VELOCITY_THRESHOLD)
                
                if not moving:
                    self.time_in_bottom_zone += 1.0 / FRAME_RATE
                
                if self.time_in_bottom_zone > DEAD_TIME_THRESHOLD and self.state != 'DEAD':
                    self.state = 'DEAD'
                else:
                    self.time_in_bottom_zone = 0
            
            if self.time_sinking_with_movement > 5 and self.state not in ['DEAD', 'SICK']:
                self.state = 'SICK'
        else:
            self.time_in_bottom_zone = 0
            self.time_sinking_with_movement = 0
            if self.state == 'DEAD':
                self.state = 'HEALTHY'

        if self.state != 'DEAD':
            if current_y < frame_h * TOP_ZONE_THRESHOLD:
                self.time_in_top_zone += 1.0 / FRAME_RATE
                if self.time_in_top_zone > 5 and self.state != 'SICK':
                    self.state = 'SICK'
            else:
                self.time_in_top_zone = 0

        if self.state not in ['DEAD', 'SICK']:
            self.state = 'HEALTHY'

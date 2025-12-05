#!/usr/bin/env python3

import gi
import cv2
import numpy as np
import threading
import queue
import time

gi.require_version("Aravis", "0.8")
from gi.repository import Aravis, GLib

class CameraThread(threading.Thread):
    def __init__(self, ip, name, stop_event, exposure_time=20000, fps=15, pixel_format='BayerRG8'):
        threading.Thread.__init__(self)
        self.ip = ip
        self.name = name
        self.image_queue = queue.Queue(maxsize=2)
        self.stop_event = stop_event
        self.exposure_time = exposure_time
        self.fps_limit = fps
        self.desired_pixel_format = pixel_format
        self.cam = None
        self.stream = None
        self.frame_counter = 0
        self.fps_data = {
            'last_time': time.time(),
            'frame_count': 0,
            'fps': 0
        }
        self.timeout_count = 0

    def run(self):
        print(f"[{self.name}] Starting camera thread for {self.ip}")

        # Retry loop for initialization
        for attempt in range(5):
            try:
                self.cam = Aravis.Camera.new(self.ip)
                if self.cam:
                    print(f"[{self.name}] ✓ Connected to camera: {self.ip}")
                    break
            except Exception as e:
                print(f"[{self.name}] Connection attempt {attempt+1} failed: {e}")
            
            if attempt < 4:
                print(f"[{self.name}] Retrying in 2s...")
                time.sleep(2)

        if self.cam is None:
            print(f"[{self.name}] FATAL: Could not connect to camera {self.ip} after 5 attempts.")
            return

        try:
            self.cam.set_string("AcquisitionMode", "Continuous")
            self.cam.set_string("TriggerMode", "Off")
            self.cam.set_integer("GevHeartbeatTimeout", 5000)
            
            try:
                self.cam.set_boolean("AcquisitionFrameRateEnable", True)
                self.cam.set_float("AcquisitionFrameRate", float(self.fps_limit))
            except Exception as e:
                print(f"[{self.name}] Warn: Could not set FPS limit: {e}")

            try:
                self.cam.set_string("PixelFormat", self.desired_pixel_format)
            except Exception as e:
                print(f"[{self.name}] Warn: {self.desired_pixel_format} not supported, trying fallback formats")
                fallback_formats = ['RGB8Packed', 'RGB8', 'BayerRG8']
                for fmt in fallback_formats:
                    if fmt == self.desired_pixel_format: continue
                    try:
                        self.cam.set_string("PixelFormat", fmt)
                        break
                    except: pass

            try:
                self.cam.set_float("ExposureTime", float(self.exposure_time))
            except Exception:
                try:
                    self.cam.set_float("ExposureTimeAbs", float(self.exposure_time))
                except Exception as e:
                    print(f"[{self.name}] Warning: Could not set exposure time: {e}")

            self.pixel_format = self.cam.get_pixel_format_as_string()
            try:
                self.width = self.cam.get_integer("Width")
                self.height = self.cam.get_integer("Height")
            except Exception as e:
                self.width = 1920
                self.height = 1080

            try:
                self.cam.set_integer("GevSCPSPacketSize", 1500)
            except Exception as e:
                print(f"[{self.name}] Warning: Could not set packet size: {e}")
            
            try:
                self.cam.set_integer("GevSCPD", 10000)
            except Exception as e:
                print(f"[{self.name}] Warning: Could not set inter-packet delay: {e}")

            self.stream = self.cam.create_stream(None, None)
            self.stream.set_property("packet-resend", 1)
            self.stream.set_property("socket-buffer-size", 64 * 1024 * 1024)

            self.payload = self.cam.get_payload()
            for _ in range(30):
                self.stream.push_buffer(Aravis.Buffer.new_allocate(self.payload))

            self.cam.start_acquisition()
            print(f"[{self.name}] Acquisition started")
            
            while not self.stop_event.is_set():
                try:
                    buffer = self.stream.timeout_pop_buffer(1000000)

                    if buffer:
                        if buffer.get_status() == Aravis.BufferStatus.SUCCESS:
                            self.frame_counter += 1
                            data = buffer.get_data()
                            
                            if data:
                                frame_data = np.frombuffer(data, dtype=np.uint8)
                                buffer_status = True
                                if self.pixel_format == "BayerRG8":
                                    expected_size = self.width * self.height
                                    if len(frame_data) >= expected_size:
                                        bayer_img = frame_data[:expected_size].reshape(self.height, self.width)
                                        frame_bgr = cv2.cvtColor(bayer_img, cv2.COLOR_BAYER_BG2BGR)
                                    else:
                                        buffer_status = False
                                elif self.pixel_format in ["RGB8", "RGB8Packed"]:
                                    expected_size = self.width * self.height * 3
                                    if len(frame_data) >= expected_size:
                                        frame_rgb = frame_data[:expected_size].reshape(self.height, self.width, 3)
                                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                                    else:
                                        buffer_status = False
                                elif self.pixel_format == "Mono8":
                                    expected_size = self.width * self.height
                                    if len(frame_data) >= expected_size:
                                        mono_img = frame_data[:expected_size].reshape(self.height, self.width)
                                        frame_bgr = cv2.cvtColor(mono_img, cv2.COLOR_GRAY2BGR)
                                    else:
                                        buffer_status = False
                                else:
                                    buffer_status = False
                                
                                if buffer_status and 'frame_bgr' in locals():
                                    current_time = time.time()
                                    self.fps_data['frame_count'] += 1
                                    if current_time - self.fps_data['last_time'] >= 1.0:
                                        self.fps_data['fps'] = self.fps_data['frame_count'] / (current_time - self.fps_data['last_time'])
                                        self.fps_data['frame_count'] = 0
                                        self.fps_data['last_time'] = current_time

                                    try:
                                        self.image_queue.put_nowait(frame_bgr.copy())
                                    except queue.Full:
                                        try:
                                            self.image_queue.get_nowait()
                                            self.image_queue.put_nowait(frame_bgr.copy())
                                        except: pass
                        else:
                            self.timeout_count += 1
                    if buffer:
                        self.stream.push_buffer(buffer)
                except Exception as e:
                    pass

        except Exception as e:
            print(f"[{self.name}] Camera initialization failed: {e}")
        finally:
            self.cleanup()

    def get_frame(self):
        try:
            latest_frame = None
            while True:
                try:
                    latest_frame = self.image_queue.get_nowait()
                except queue.Empty:
                    break
            return latest_frame
        except Exception as e:
            return None

    def get_frame_with_overlay(self):
        return self.get_frame()

    def get_stats(self):
        return {
            'name': self.name,
            'ip': self.ip,
            'frames_captured': self.frame_counter,
            'fps': self.fps_data['fps'],
            'queue_size': self.image_queue.qsize(),
            'timeouts': self.timeout_count
        }

    def cleanup(self):
        try:
            if self.cam:
                self.cam.stop_acquisition()
                time.sleep(0.1)
                if self.stream: self.stream = None
                self.cam = None
        except Exception as e:
            print(f"[{self.name}] Cleanup error: {e}")

class TopViewCamera(CameraThread):
    def __init__(self, stop_event, device_id, exposure_time=20000, fps=15, pixel_format='RGB8Packed'):
        super().__init__(device_id, "TopView", stop_event, exposure_time, fps, pixel_format)

class SideViewCamera(CameraThread):
    def __init__(self, stop_event, device_id, exposure_time=20000, fps=15, pixel_format='RGB8Packed'):
        super().__init__(device_id, "SideView", stop_event, exposure_time, fps, pixel_format)

class CameraManager:
    def __init__(self):
        self.stop_event = threading.Event()
        self.cameras = {}

    def add_camera(self, camera_class, name, **kwargs):
        camera = camera_class(self.stop_event, **kwargs)
        camera.daemon = True
        self.cameras[name] = camera

    def start_all(self):
        for name, camera in self.cameras.items():
            camera.start()

    def stop_all(self):
        self.stop_event.set()
        for name, camera in self.cameras.items():
            camera.join(timeout=1.0)

    def get_frame(self, camera_name):
        if camera_name in self.cameras:
            return self.cameras[camera_name].get_frame()
        return None

    def get_frame_with_overlay(self, camera_name):
        if camera_name in self.cameras:
            return self.cameras[camera_name].get_frame_with_overlay()
        return None

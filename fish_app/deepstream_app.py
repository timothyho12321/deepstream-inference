import sys
import time
import gi
import numpy as np
import threading
import yaml
import os
import cv2
from collections import deque

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Try to import pyds
try:
    import pyds
except ImportError:
    print("pyds not found. Make sure you are running in DeepStream container.")
    # For development outside container, we might want to mock or just warn
    # sys.exit(1)

from frame_manager import FrameManager
from camera_thread import CameraManager, TopViewCamera, SideViewCamera
from fish_tracker import FishTracker

# Initialize GStreamer
Gst.init(None)

class FishDeepStreamApp:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.frame_manager = FrameManager()
        self.cam_manager = CameraManager()
        self.pipeline = None
        self.loop = None
        self.stop_event = threading.Event()
        
        self.fish_db = {} # track_id -> FishTracker
        
        # Initialize cameras
        self._init_cameras()
        
        # Initialize DeepStream Pipeline
        self._init_pipeline()

    def _init_cameras(self):
        video_cfg = self.config.get('video', {})
        top_src = video_cfg.get('top_source')
        side_src = video_cfg.get('side_source')
        exposure = float(video_cfg.get('exposure_time', 20000))
        fps = int(video_cfg.get('fps', 30))
        pixel_fmt = video_cfg.get('pixel_format', 'BayerRG8')

        if top_src:
            self.cam_manager.add_camera(TopViewCamera, 'TopView', device_id=top_src, exposure_time=exposure, fps=fps, pixel_format=pixel_fmt)
        if side_src:
            self.cam_manager.add_camera(SideViewCamera, 'SideView', device_id=side_src, exposure_time=exposure, fps=fps, pixel_format=pixel_fmt)
        
        self.cam_manager.start_all()

    def _init_pipeline(self):
        # Create GStreamer pipeline
        self.pipeline = Gst.Pipeline()
        
        # Stream Muxer
        streammux = Gst.ElementFactory.make("nvstreammux", "Stream-Muxer")
        streammux.set_property('width', 960)
        streammux.set_property('height', 720)
        streammux.set_property('batch-size', 2)
        streammux.set_property('batched-push-timeout', 40000)
        self.pipeline.add(streammux)
        
        # Sources (AppSrc)
        self.appsrc_top = self._create_source(streammux, 0, "source_top")
        self.appsrc_side = self._create_source(streammux, 1, "source_side")
        
        # Inference (nvinfer)
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        pgie.set_property('config-file-path', "nvdsinfer_config.txt")
        self.pipeline.add(pgie)
        
        # Tracker
        tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        tracker.set_property('ll-lib-file', '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so')
        tracker.set_property('ll-config-file', '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml')
        tracker.set_property('enable-batch-process', 1)
        tracker.set_property('enable-past-frame', 1)
        self.pipeline.add(tracker)

        # Tiler (Combine 2 streams into 1 for display/streaming)
        tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
        tiler.set_property("rows", 1)
        tiler.set_property("columns", 2)
        tiler.set_property("width", 1920)
        tiler.set_property("height", 720)
        self.pipeline.add(tiler)
        
        # Converter & OSD
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvideo-converter")
        nvosd = Gst.ElementFactory.make("nvdsosd", "nv-onscreendisplay")
        self.pipeline.add(nvvidconv)
        self.pipeline.add(nvosd)
        
        # Sink (AppSink to get frames back to python for streaming)
        # We need to convert to RGBA for appsink
        nvvidconv_sink = Gst.ElementFactory.make("nvvideoconvert", "nvvideo-converter-sink")
        capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
        caps = Gst.Caps.from_string("video/x-raw, format=RGBA")
        capsfilter.set_property("caps", caps)
        
        appsink = Gst.ElementFactory.make("appsink", "app-sink")
        appsink.set_property("emit-signals", True)
        appsink.connect("new-sample", self._on_new_sample, None)
        
        self.pipeline.add(nvvidconv_sink)
        self.pipeline.add(capsfilter)
        self.pipeline.add(appsink)
        
        # Link elements
        # streammux -> pgie -> tracker -> tiler -> nvvidconv -> nvosd -> nvvidconv_sink -> capsfilter -> appsink
        
        streammux.link(pgie)
        pgie.link(tracker)
        tracker.link(tiler)
        tiler.link(nvvidconv)
        nvvidconv.link(nvosd)
        nvosd.link(nvvidconv_sink)
        nvvidconv_sink.link(capsfilter)
        capsfilter.link(appsink)
        
        # Add probe to get metadata (Fish Logic)
        # We probe at OSD sink pad to get metadata before drawing? 
        # Or after tracker?
        # If we probe at OSD sink pad, we can modify display text.
        osd_sink_pad = nvosd.get_static_pad("sink")
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, self._osd_sink_pad_buffer_probe, None)

    def _create_source(self, streammux, pad_index, name):
        source = Gst.ElementFactory.make("appsrc", name)
        source.set_property("is-live", True)
        source.set_property("format", 3) # GST_FORMAT_TIME
        
        # Caps
        caps = Gst.Caps.from_string("video/x-raw,format=BGR,width=960,height=720,framerate=30/1")
        source.set_property("caps", caps)
        
        videoconvert = Gst.ElementFactory.make("videoconvert", f"convert_{name}")
        nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", f"nvconvert_{name}")
        
        self.pipeline.add(source)
        self.pipeline.add(videoconvert)
        self.pipeline.add(nvvideoconvert)
        
        source.link(videoconvert)
        videoconvert.link(nvvideoconvert)
        
        sinkpad = streammux.get_request_pad(f"sink_{pad_index}")
        srcpad = nvvideoconvert.get_static_pad("src")
        srcpad.link(sinkpad)
        
        return source

    def _osd_sink_pad_buffer_probe(self, pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        try:
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        except NameError:
            # pyds not imported
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            # Determine view type based on source_id
            # source_id 0 -> Top, 1 -> Side
            view_type = 'top' if frame_meta.source_id == 0 else 'side'
            
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                
                track_id = obj_meta.object_id
                
                # Update FishTracker
                if track_id not in self.fish_db:
                    self.fish_db[track_id] = FishTracker(track_id, view_type, self.config)
                
                tracker = self.fish_db[track_id]
                rect = obj_meta.rect_params
                tracker.update([rect.left, rect.top, rect.width, rect.height], obj_meta.class_id)
                tracker.check_behavior()
                
                # Update OSD text
                if tracker.state != 'HEALTHY':
                    obj_meta.text_params.display_text = f"{obj_meta.text_params.display_text} {tracker.state}"
                    # Change color if possible (DeepStream python bindings for color are tricky)
                    # obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 1.0) # Red for dead/sick
                
                try: 
                    l_obj = l_obj.next
                except StopIteration:
                    break
            
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
                
        return Gst.PadProbeReturn.OK

    def _on_new_sample(self, sink, data):
        # Get frame from appsink and update FrameManager
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        
        # Extract buffer to numpy
        # The buffer is RGBA, 1920x720 (tiled side-by-side)
        # We need to convert it to numpy array
        
        # Get width and height from caps
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        
        # Get data
        buffer = buf.extract_dup(0, buf.get_size())
        arr = np.ndarray(
            (height, width, 4),
            buffer=buffer,
            dtype=np.uint8
        )
        
        # Convert RGBA to BGR for FrameManager/OpenCV
        frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        
        # Split the tiled image back into two frames for the FrameManager
        # Tiler configured as 1 row, 2 columns. Width 1920, Height 720.
        # So each frame is 960x720.
        
        mid_point = width // 2
        frame_top = frame_bgr[:, :mid_point]
        frame_side = frame_bgr[:, mid_point:]
        
        self.frame_manager.update_frame1(frame_top)
        self.frame_manager.update_frame2(frame_side)
        
        return Gst.FlowReturn.OK

    def feed_cameras(self):
        # Thread to feed frames from CameraManager to AppSrc
        while not self.stop_event.is_set():
            # Top View
            frame_top = self.cam_manager.get_frame('TopView')
            if frame_top is not None:
                # Push to appsrc_top
                data = frame_top.tobytes()
                buf = Gst.Buffer.new_allocate(None, len(data), None)
                buf.fill(0, data)
                self.appsrc_top.emit("push-buffer", buf)
            
            # Side View
            frame_side = self.cam_manager.get_frame('SideView')
            if frame_side is not None:
                # Push to appsrc_side
                data = frame_side.tobytes()
                buf = Gst.Buffer.new_allocate(None, len(data), None)
                buf.fill(0, data)
                self.appsrc_side.emit("push-buffer", buf)
            
            time.sleep(0.01)

    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        
        # Start feeding thread
        feed_thread = threading.Thread(target=self.feed_cameras)
        feed_thread.start()
        
        # Start Main Loop
        self.loop = GLib.MainLoop()
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_event.set()
            feed_thread.join()
            self.pipeline.set_state(Gst.State.NULL)
            self.cam_manager.stop_all()

if __name__ == "__main__":
    app = FishDeepStreamApp()
    app.run()

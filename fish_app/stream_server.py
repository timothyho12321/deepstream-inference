import cv2
import numpy as np
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from frame_manager import FrameManager

# ==========================================
# Fallback Generators (if camera fails)
# ==========================================
class FallbackGenerator:
    def __init__(self, text="NO SIGNAL", color=(0, 0, 255)):
        self.text = text
        self.color = color
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def get_frame(self):
        self.frame.fill(0)
        # Draw background
        cv2.rectangle(self.frame, (0,0), (640, 480), self.color, 2)
        # Draw Text
        cv2.putText(self.frame, self.text, (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Add timestamp
        t_str = time.strftime('%H:%M:%S', time.localtime())
        cv2.putText(self.frame, t_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        return self.frame

# ==========================================
# HTTP Handler
# ==========================================
class DualStreamHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, frame_manager=None, **kwargs):
        self.frame_manager = frame_manager if frame_manager else FrameManager()
        self.fallback_top = FallbackGenerator("TOP CAM OFFLINE", (0, 0, 100))
        self.fallback_side = FallbackGenerator("SIDE CAM OFFLINE", (0, 0, 100))
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        pass  # Suppress console logs to keep output clean

    def do_GET(self):
        if self.path == '/stream1':
            self._serve_stream(1)
        elif self.path == '/stream2':
            self._serve_stream(2)
        elif self.path == '/both':
            self._serve_combined()
        elif self.path == '/':
            self._serve_ui()
        elif self.path == '/status':
            self._serve_status()
        else:
            self.send_error(404)

    def _serve_stream(self, stream_id):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
        self.end_headers()

        try:
            while True:
                if stream_id == 1:
                    frame, _, _ = self.frame_manager.get_frame1()
                    fallback = self.fallback_top
                else:
                    frame, _, _ = self.frame_manager.get_frame2()
                    fallback = self.fallback_side

                if frame is None:
                    frame = fallback.get_frame()
                    time.sleep(0.1) # Slow down if no source

                # Encode
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                self.wfile.write(b'--FRAME\r\n')
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg.tobytes())
                self.wfile.write(b'\r\n')

                # Cap streaming FPS to ~25 to save bandwidth
                time.sleep(0.01)

        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_combined(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
        self.end_headers()

        try:
            while True:
                f1, _, _ = self.frame_manager.get_frame1()
                f2, _, _ = self.frame_manager.get_frame2()

                if f1 is None: f1 = self.fallback_top.get_frame()
                if f2 is None: f2 = self.fallback_side.get_frame()

                # Resize to match height if needed
                if f1.shape[0] != f2.shape[0]:
                    f2 = cv2.resize(f2, (int(f2.shape[1] * (f1.shape[0]/f2.shape[0])), f1.shape[0]))

                combined = np.hstack((f1, f2))

                # Encode
                _, jpeg = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 80])

                self.wfile.write(b'--FRAME\r\n')
                self.send_header('Content-Type', 'image/jpeg')
                self.end_headers()
                self.wfile.write(jpeg.tobytes())
                self.wfile.write(b'\r\n')
                time.sleep(0.05) # 20 FPS for combined

        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_ui(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        html = """
        <html>
        <body style="background:#111; color:#ddd; font-family:sans-serif; text-align:center;">
            <h1>Fish Activity Monitoring System</h1>
            <div style="display:flex; justify-content:center; flex-wrap:wrap; gap:10px;">
                <div><h3>Top View</h3><img src="/stream1" style="width:640px; border:2px solid #0ff;"></div>
                <div><h3>Side View</h3><img src="/stream2" style="width:640px; border:2px solid #f00;"></div>
            </div>
            <br>
            <a href="/both" style="color:#0ff; font-size:20px;">View Combined Stream</a>
        </body>
        </html>
        """
        self.wfile.write(html.encode('utf-8'))

    def _serve_status(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        info = self.frame_manager.get_frame_info()
        self.wfile.write(str(info).encode('utf-8'))

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def start_stream_server(frame_manager, port=8000):
    server = ThreadedHTTPServer(('0.0.0.0', port),
                                lambda *args, **kwargs: DualStreamHandler(*args, frame_manager=frame_manager, **kwargs))

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server

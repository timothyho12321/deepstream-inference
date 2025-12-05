import threading
import time
import cv2

class FrameManager:
    """
    Thread-safe Singleton to share frames between the Detection Loop (Producer)
    and the Streaming Server (Consumer).
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        # Stream 1 corresponds to TOP View
        # Stream 2 corresponds to SIDE View
        self._frame1 = None
        self._frame2 = None

        self._frame1_lock = threading.Lock()
        self._frame2_lock = threading.Lock()

        self._frame1_time = 0
        self._frame2_time = 0
        self._frame1_count = 0
        self._frame2_count = 0

    def update_frame1(self, new_frame):
        """Update Top View Frame"""
        with self._frame1_lock:
            if new_frame is not None:
                self._frame1 = new_frame.copy()
            else:
                self._frame1 = None
            self._frame1_time = time.time()
            self._frame1_count += 1

    def update_frame2(self, new_frame):
        """Update Side View Frame"""
        with self._frame2_lock:
            if new_frame is not None:
                self._frame2 = new_frame.copy()
            else:
                self._frame2 = None
            self._frame2_time = time.time()
            self._frame2_count += 1

    def get_frame1(self):
        """Get Top View Frame"""
        with self._frame1_lock:
            if self._frame1 is not None:
                return self._frame1.copy(), self._frame1_time, self._frame1_count
        return None, 0, 0

    def get_frame2(self):
        """Get Side View Frame"""
        with self._frame2_lock:
            if self._frame2 is not None:
                return self._frame2.copy(), self._frame2_time, self._frame2_count
        return None, 0, 0

    def get_frame_info(self):
        """Get metadata for status endpoints"""
        with self._frame1_lock:
            with self._frame2_lock:
                return {
                    'stream1_top': {
                        'active': self._frame1 is not None,
                        'count': self._frame1_count,
                        'timestamp': self._frame1_time
                    },
                    'stream2_side': {
                        'active': self._frame2 is not None,
                        'count': self._frame2_count,
                        'timestamp': self._frame2_time
                    }
                }

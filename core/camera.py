"""
Camera Module for VMS Dashboard.
Implements threaded video frame capture with strict instant hardware release hooks.
"""
import threading
import time
import logging
import gc
from collections import deque
import cv2

class VideoStream:
    """Class to capture frames from camera/video source in a background thread."""

    def __init__(self, src=0, width=None, height=None):
        self.src = src
        # Convert source string to integer if it represents a camera index
        try:
            self.src = int(src)
        except ValueError:
            self.src = src

        self.stream = cv2.VideoCapture(self.src)
        self.width = width
        self.height = height

        self.stopped = threading.Event()
        self.read_lock = threading.Lock()
        self.frame = None
        self.grabbed = False

        if not self.stream.isOpened():
            logging.error("Failed to open camera/video source: %s", src)
            raise IOError(f"Cannot open video source: {src}")

        self.grabbed, self.frame = self.stream.read()
        logging.info("VideoStream successfully initialized for source: %s", src)

    def start(self):
        """Starts the thread to read frames from the video stream."""
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self) -> None:
        """Continuously reads frames from the stream until stopped."""
        while not self.stopped.is_set():
            if not self.grabbed:
                self.stop()
                break

            grabbed, frame = self.stream.read()

            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

            # Avoid pegging CPU when reading fast
            time.sleep(0.005)

    def read(self):
        """Returns the most recent frame."""
        with self.read_lock:
            frame = self.frame.copy() if self.frame is not None else None
            grabbed = self.grabbed

        if grabbed and frame is not None:
            if self.width is not None and self.height is not None:
                frame = cv2.resize(frame, (self.width, self.height))

        return grabbed, frame

    def stop(self) -> None:
        """Stops the thread, releases camera, and forces garbage collection."""
        self.stopped.set()
        
        # Wait for update thread to join
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        # Release cv2 resource immediately
        if self.stream.isOpened():
            self.stream.release()

        with self.read_lock:
            self.frame = None
            self.grabbed = False

        # Force garbage collection to instantly release device handle on Windows
        gc.collect()
        logging.info("VideoStream thread stopped, device released, and garbage collected.")


class FPSCalculator:
    """Calculates running average FPS over a sliding window."""

    def __init__(self, window_size: int = 30):
        self.times = deque(maxlen=window_size)

    def tick(self) -> None:
        """Appends current timestamp to calculate moving framerate."""
        self.times.append(time.time())

    def get_fps(self) -> float:
        """Computes current average FPS."""
        if len(self.times) < 2:
            return 0.0
        elapsed = self.times[-1] - self.times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.times) - 1) / elapsed

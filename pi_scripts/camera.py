from picamera import PiCamera
import time
import os
import tempfile
import itertools as IT

# Create unique filename (add one to previous filename)
def uniquify(path, sep = ''):
    def name_sequence():
        count = IT.count()
        yield ''
        while True:
            yield '{s}{n:d}'.format(s = sep, n = next(count))
    orig = tempfile._name_sequence 
    with tempfile._once_lock:
        tempfile._name_sequence = name_sequence()
        path = os.path.normpath(path)
        dirname, basename = os.path.split(path)
        filename, ext = os.path.splitext(basename)
        fd, filename = tempfile.mkstemp(dir = dirname, prefix = filename, suffix = ext)
        tempfile._name_sequence = orig
    os.remove(filename)
    return filename

# Initialize camera
camera = PiCamera()
camera.resolution = (640, 480)

#camera.start_preview()
time.sleep(2)

path = "/home/pi/Desktop/captures"
#filename = "my_photo.jpg"
filename = "video.h264"

# Main loop - Generate one output video file for each iteration
for n in range(100):
    unique_filename = uniquify(os.path.join(path, filename))

    #print(unique_filename)
    #camera.capture(unique_filename)

    camera.start_recording(unique_filename)
    time.sleep(60*5)
    camera.stop_recording()

#camera.stop_preview()

# Lint as: python3
# Copyright 2021 Neil Tender
#
r"""Use PyCoral to detect objects in a given video.

To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

Example usage:
```
without tracking:
python3 process_video.py --tracking 0  --model models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite --labels models/coco_labels.txt --input test_data/input_data/video56.m4v --output test_data/output_data/video56_processed.m4v

with tracking:
python3 process_video.py --tracking 1  --model models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite --labels models/coco_labels.txt --input test_data/input_data/video56.m4v --output test_data/output_data/video56_processed_tracking.m4v

```
"""

import argparse
import time
import copy
import cv2

import numpy as np

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

import sort_tracker

# Draw box around detected object with label
def draw_detection(image, pt_left_top, pt_right_bot, label_string, color_background, color_foreground):
  image_width = image.shape[1]
  image_height = image.shape[0]

  text_fontFace=cv2.FONT_HERSHEY_SIMPLEX
  text_fontScale=0.5
  text_thickness=1
  
  text_size = cv2.getTextSize(text=label_string,
                              fontFace=text_fontFace,
                              fontScale=text_fontScale,
                              thickness=text_thickness)
  text_baseLine=text_size[1]
  text_width=text_size[0][0]
  text_height=text_size[0][1]

  # Place text and textbox whereever it will fit
  # First try upper left
  textbox_top=pt_left_top[1]-text_height-text_baseLine
  textbox_bot=pt_left_top[1]
  textbox_left=pt_left_top[0]
  textbox_right=pt_left_top[0]+text_width
  text_origin=(pt_left_top[0],pt_left_top[1]-text_baseLine)
  if ((textbox_top < 0) or (textbox_left < 0)):
      # try lower left
      textbox_top=pt_right_bot[1]
      textbox_bot=pt_right_bot[1]+text_height+text_baseLine
      textbox_left=pt_left_top[0]
      textbox_right=pt_left_top[0]+text_width
      text_origin=(pt_left_top[0],pt_right_bot[1]+text_height)
  if ((textbox_bot >= image_height) or (textbox_right >= image_width)):
      # try upper right
      textbox_top=pt_left_top[1]-text_height-text_baseLine
      textbox_bot=pt_left_top[1]
      textbox_left=pt_right_bot[0]-text_width
      textbox_right=pt_right_bot[0]
      text_origin=(pt_right_bot[0]-text_width,pt_left_top[1]-text_baseLine)
  if ((textbox_top < 0) or (textbox_left < 0)):
      # try lower right
      textbox_top=pt_right_bot[1]
      textbox_bot=pt_right_bot[1]+text_height+text_baseLine
      textbox_left=pt_right_bot[0]-text_width
      textbox_right=pt_right_bot[0]
      text_origin=(pt_right_bot[0]-text_width,pt_right_bot[1]+text_height)
    
  cv2.rectangle(img=image,
                pt1=(textbox_left,textbox_top),
                pt2=(textbox_right,textbox_bot),
                color=color_background,
                thickness=cv2.FILLED)

  cv2.putText(img=image,
              text=label_string,
              org=(text_origin),
              fontFace=text_fontFace,
              fontScale=text_fontScale,
              thickness=2,
              color=(0,0,0))

  cv2.rectangle(img=image,
                pt1=pt_left_top,
                pt2=pt_right_bot,
                color=color_foreground,
                thickness=1)

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', required=True,
                      help='File path of .tflite file')
  parser.add_argument('-i', '--input', required=True,
                      help='File path of image to process')
  parser.add_argument('-l', '--labels', help='File path of labels file')
  parser.add_argument('-t', '--threshold', type=float, default=0.55,
                      help='Score threshold for detected objects')
  parser.add_argument('-o', '--output', required=True,
                      help='File path for the result image with annotations')
  parser.add_argument("--tracking", help="0=no tracking, 1=tracking.", type=int, default=1)
  parser.add_argument("--max_age", 
                      help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=3)
  parser.add_argument("--min_hits", 
                      help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
  parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.4)
  parser.add_argument("--approach_tracking_depth", help="Approach tracking depth.", type=int, default=25)
  parser.add_argument("--approach_tracking_threshold", help="Approach tracking threshold.", type=float, default=4.0)

    
  args = parser.parse_args()

  labels = read_label_file(args.labels) if args.labels else {}
  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  input_filename = args.input
  output_filename = args.output
  begin_frame_number = 0
  end_frame_number = -1
  no_display = 0

  # Create a VideoCapture object and read from input file
  # If the input is the camera, pass 0 instead of the video file name
  capture = cv2.VideoCapture(input_filename)

  # Check if video or camera opened successfully
  if (capture.isOpened()== False): 
      print("Error opening video stream or file")
      exit()
 
  # Retrieve information about video and jump to begin_frame_number and 
  total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
  frame_rate = capture.get(cv2.CAP_PROP_FPS)
  capture.set(cv2.CAP_PROP_POS_FRAMES, begin_frame_number)
  frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
  # If output filename is provided, create video writer object
  if (output_filename != ""):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter(output_filename,fourcc,frame_rate,(frame_width,frame_height))

  # Setup tracker
  mot_tracker = sort_tracker.Sort(max_age=args.max_age, 
                                  min_hits=args.min_hits,
                                  iou_threshold=args.iou_threshold) #create instance of the SORT tracker
  all_tracks = {}
  
  # Read until video is completed
  time_start = time.time()
  frame_count = 0

  while (frame_count < total_frames):

    # Capture frame-by-frame
    current_frame_number = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    current_timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
    current_frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    current_frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Capture one frame
    ret, frame = capture.read()
#    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if ((ret == True) and ((current_frame_number <= end_frame_number) or (end_frame_number == -1))):
      frame_count = frame_count + 1
      time_elapsed = time.time() - time_start
      processing_speed = frame_count / time_elapsed
      print("frame count {:d} / {:d}, timestamp {:0.1f}ms, dimensions {:d}x{:d}, frame rate {:0.2f}fps, elapsed time {:0.1f}sec, frame_count {:d}, processing speed (frames/sec) {:0.2f}".format(frame_count, total_frames, current_timestamp, current_frame_width, current_frame_height, frame_rate, time_elapsed, current_frame_number, processing_speed))

      # Perform inference
      _, scale = common.set_resized_input(
          interpreter, (current_frame_width, current_frame_height), lambda size: cv2.resize(frame, size))
      start = time.perf_counter()
      interpreter.invoke()
      inference_time = time.perf_counter() - start
      objs = detect.get_objects(interpreter, args.threshold, scale)
      print('inference time = %.2f ms' % (inference_time * 1000))

      # Filter results to only objects of interest.
      if (args.tracking == 1):
        objects_of_interest = [2, 3, 5, 7]  #  2=car  3=motorcycle  5=bus  7=truck
      else:
        objects_of_interest = [1, 2, 3, 5, 7]  #  1=bicycle  2=car  3=motorcycle  5=bus  7=truck

      objs_filtered = []
      for obj in objs:
        if (obj.id in objects_of_interest):
          objs_filtered.append(obj)

      # Update trackers
      dets = np.array([[obj.bbox.xmin,obj.bbox.ymin,obj.bbox.xmax,obj.bbox.ymax,obj.score]  for obj in objs_filtered])
      if len(objs_filtered) > 0:
        trackers = mot_tracker.update(dets)
      else:
        trackers = np.array([])

      # Generate output image (with overlayed object rectangles) if it will be needed.
      if ((not no_display) or (output_filename != "")):

        # Copy the image
        image = copy.deepcopy(frame)

        if (args.tracking == 0):
          for obj in objs_filtered:
            bbox = obj.bbox
            pt_left_top = (bbox.xmin, bbox.ymin)
            pt_right_bot = (bbox.xmax, bbox.ymax)
            label_string = '%s %.2f' % (labels.get(obj.id, obj.id), obj.score)
            color_foreground = (255, 0, 0)
            color_background = (255, 255, 255)
            draw_detection(image, pt_left_top, pt_right_bot, label_string, color_background, color_foreground)
        else:
          # Determine if vehicle is approaching by checking if box sizes have been increasing.
          any_approaching_vehicle = False
          for tracker in trackers:
            pt_left_top = (int(round(tracker[0])), int(round(tracker[1])))
            pt_right_bot = (int(round(tracker[2])), int(round(tracker[3])))
            track_id = tracker[4]

            if track_id not in all_tracks:
              all_tracks[track_id] = []
            all_tracks[track_id].append([pt_left_top, pt_right_bot])

            m_w = 0.0
            m_h = 0.0
            if (len(all_tracks[track_id]) < args.approach_tracking_depth):
              approaching_vehicle = False;
            else:
              last_N_widths  = np.array([(bbox[1][0] - bbox[0][0]) for bbox in all_tracks[track_id][-args.approach_tracking_depth:]])
              last_N_heights = np.array([(bbox[1][1] - bbox[0][1]) for bbox in all_tracks[track_id][-args.approach_tracking_depth:]])
              average_height = np.mean(last_N_heights)

              # perform linear regression on last N widths and heights
              A = np.vstack([range(args.approach_tracking_depth), np.ones(args.approach_tracking_depth)]).T
              m_w, c_w = np.linalg.lstsq(A, last_N_widths, rcond=None)[0]
              m_h, c_h = np.linalg.lstsq(A, last_N_heights, rcond=None)[0]

              m_h_weighted = m_h * average_height
              m_w_weighted = m_w * average_height

              # Compare against threshold to determine if vehicle is approaching
              approaching_vehicle = (m_h_weighted > args.approach_tracking_threshold) and \
                                    (((m_w_weighted > args.approach_tracking_threshold) and (m_w_weighted > 0.5*m_h_weighted)) or (m_w_weighted < -10.0*m_h_weighted))

            # Draw box around detection.
            label_string = '%d' % (track_id)
            if (approaching_vehicle):
              any_approaching_vehicle = True
              color_foreground = (0, 0, 255)
            else:
              color_foreground = (0, 255, 0)
            color_background = (255, 255, 255)
            draw_detection(image, pt_left_top, pt_right_bot, label_string, color_background, color_foreground)

          # Draw red box around frame if vehicle is approaching.
          if (any_approaching_vehicle):
            cv2.rectangle(img=image,
                          pt1=(0, 0),
                          pt2=(frame_width, frame_height),
                          color=(0, 0, 255),
                          thickness=30)
        frame_out = image
#        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        video_out.write(frame_out)

    # Break the loop
    else: 
        break

  # When everything is done, release the video capture and writer objects
  capture.release()
  if (output_filename != ""):
    video_out.release()
            
  # Closes all the frames
#  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()


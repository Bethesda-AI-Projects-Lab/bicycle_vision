# CAR-BAC Bicycle Rear View Vision

CAR-BAC (Car Approaching Rearview Bicycle Alert Camera) enhances bicyclists' safety by improving situational awareness when riding in traffic with motor vehicles. It detects vehicles approaching from behind and emits an audible alert when a vehicle approach exceeds a threshold. Thus CAR-BAC can help the cyclist maintain full 360-degree awareness, even if he/she might be reacting simultaneously to threats ahead of the bicycle.

The system is built around a Raspberry Pi 4 single-board computer, powered by a LiPO battery pack. These devices are placed in the bicyclist's saddle bag. A 5 MP Arducam image sensor is connected and mounted to the seatpost facing aft. A Google Coral Edge TPU runs the SSDLite object detection algorithm on data from the video feed. Vehicle detections are passed to a SORT tracker and then a linear classifier that identifies approaching vehicle tracks. The system alerts the cyclist audibly by sounding a buzzer attached to the Pi's GPIO pin.

Here we demonstrate a viable prototype for CAR-BAC. We have entered CAR-BAC in the [Eyes on Edge: tinyML Vision Challenge](https://www.hackster.io/contests/tinyml-vision), sponsored by the [tinyML](https://www.tinyml.org/) foundation. Our paper provides a full description of the solution, and it details possible enhancements to be pursued in future iterations. Enhancements fall into three categories: size/weight/power (SWaP) reduction, algorithm improvements, and additional features.

Components of CAR-BAC system:

![CAR BACK](https://user-images.githubusercontent.com/11370301/132293023-93656827-e750-4b45-a76b-27e16b1296c9.png)

CAR-BAC fitted on bike:

![Car Back on Bike](https://user-images.githubusercontent.com/11370301/132380344-f09759d8-1d35-4645-9d0a-329aee967107.png)

Detection examples. Red boxes indicate approaching vehicles, while green boxes signify vehicles that are not approaching.

![Detection examples](https://raw.githubusercontent.com/Bethesda-AI-Projects-Lab/bicycle_vision/main/photos/three_stills_with_detects.jpg)


Directory structure:

/
CAR-BAC Final Report.docx.pdf - Detailed description of project, hardware, software, test results, lessons learned, next steps, and more.
BicycleCamBOM.xlsx            - Bill of Material
coral/
  models/                     - TFlite models for Coral TPU
  test_data/                  - Example test data
  process_video.py            - Python script to process pre-recorded video from bicycle camera (for testing)
  run_live.py                 - Python script to process live video from bicycle camera (the main application)
  sort_tracker.py             - Implements SORT tracking algorithm
  utils.py                    - Some utility functions invoked by the top-level scripts
pi_scripts/
  camera.py                   - Python scirpt to capture live video and record to file (for testing)
  start_script_on_boot.txt    - Instructions to configure Raspberry Pi to run a Python script on bootup
photos/                       - Photos of CAR-BAC
videos/                       - Demo videos of CAR-BAC


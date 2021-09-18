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

```bash
├── CAR-BAC Final Report.docx.pdf - detailed description of project 
├── BicycleCamBOM.xlsx            - bill of materials
├── coral
│   ├── models                    - TFLite models for Coral TPU
│   ├── test_data                 - example test data
│   ├── process_video.py          - script to process pre-recorded video
│   ├── run_live.py               - script to process live video (main application)
│   ├── sort_tracker.py           - implements SORT tracking algorithm
│   └── utils.py                  - utility functions invoked by top-level scripts
├── pi_scripts
│   ├── camera.py                 - script to capture live video and record to file
│   ├── start_script_on_boot.txt  - instructions to configure Pi to run script on bootup
├── photos                        - photos of CAR-BAC
├── videos                        - demo videos of CAR-BAC
├── README.md                     - this README


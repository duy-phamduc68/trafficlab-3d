```bash
  ██    ▒▒          ██▓▓    ██   
██░░▓▓██░░██        ▓▓░░  ▒▒░░██ 
  ▓▓░░▒▒░░██        ▓▓░░▒▒░░▓▓   ████████╗██████╗  █████╗ ███████╗███████╗██╗ ██████╗██╗      █████╗ ██████╗ 
    ▓▓░░░░██        ▓▓░░░░▓▓     ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██╔════╝██║██╔════╝██║     ██╔══██╗██╔══██╗
      ▓▓░░██        ▓▓░░░░▒▒        ██║   ██████╔╝███████║█████╗  █████╗  ██║██║     ██║     ███████║██████╔╝
      ██░░██        ▓▓░░▓▓░░▒▒      ██║   ██╔══██╗██╔══██║██╔══╝  ██╔══╝  ██║██║     ██║     ██╔══██║██╔══██╗
      ██░░▒▒▒▒▒▒▒▒▒▒▒▒░░  ▓▓░░██    ██║   ██║  ██║██║  ██║██║     ██║     ██║╚██████╗███████╗██║  ██║██████╔╝
        ████▒▒░░░░▒▒▓▓▓▓            ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝     ╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝╚═════╝ 
        ████  ░░░░  ▓▓▓▓         
```

---

# Welcome to TrafficLab

---

Release: v1.0

Developed by Yuk
- [Yuk's Blog](https://yuk068.github.io/)
- [Github (Casual)](https://github.com/yuk068)
- [Github (Work)](https://github.com/duy-phamduc68)

---

Complementary resources:
- Youtube Guide at [Coming Soon](https://yuk068.github.io/)
- Check out full academic report at [Coming Soon](https://yuk068.github.io/)
- Read full blog post on TrafficLab at [Coming Soon](https://yuk068.github.io/)

---

## 1. Introduction

---

TrafficLab is an end-to-end traffic analysis suite that covers:

- Calibration: Establishing a two way projection between any CCTV and its satellite map, with support for custom SVG.
- Inference: Easily swap object detection models and object tracker along with numerous kinetics and comprehensive control of arguments.
- Visualization: A "digital twin" experience with side-by-side, synchronized view of CCTV with 3D bounding boxes and satellite view with floor box, speed, and orientation.

TrafficLab puts accessibility at the forefront, with just access to mp4 CCTV footage and knowing where that location is on Google Maps, anyone can create a fancy digital twin demo that demonstrates advanced computer vision, especially for students, individual investigators, and enthusiasts who might not have access to camera calibration and synchronized high quality satellite imagery.

Get started by navigating to any tabs on the top left corner of the program.

---

## 2. TrafficLab's Functionality

---

TrafficLab functionalities are spread across 3 main tabs, you can navigate to any of these tabs without losing work on another, below are brief description of each of the tabs.

---

### 2.1. Calibration Tab

---

Calibration Tab produces G Projection JSON files (refer to the report) which helps establish a two-way projection between the CCTV and SAT (satellite) domain, it presents a comprehensive, backwards compatible stage-based calibration process comprising of the following stages:

- Phase 1: Undistort
  - Pick Stage: Quickly validate and initialize construction/reconstruction of G Projection for a given location code.
  - Lens Stage: Configure intrinsics matrix K.
  - Undistort Stage: Adjust distortion coefficients obeying the Brown-Conrady distortion model (5 coefficients).
  - Validation 1: You can confirm the distortion and intrinsics, concluding the Phase.
- Phase 2: Homography
  - Homography Anchors Stage: Manual pair point based homography computation with RANSAC solver.
  - Homography FOV Stage: Check the warped CCTV overlaid on the SAT map, which also doubles up as a FOV polygon for intuitive visualization.
  - Validation 2: Click a ground contact point in CCTV and see it shows up on SAT map.
- Phase 3: Parallax
  - Parallax Subjects Stage: Establish Head and Ground Contact point of 2 Subjects, input their height, calculate the camera's position.
  - Distance Reference Stage: Enter the distance (obtainable from Google Maps/Earth) to establish pixel per meter ratio.
  - Validation 3: Click head point, enter height, see ground contact point in CCTV and actual position on SAT map.
- Optional:
  - SVG Stage: Compute affine matrix between SVG and SAT.
  - ROI: Choose a discard strategy for ROI.
- Final Validation: Test how 2D bounding box converts to 3D box in CCTV and floor box in SAT.
- Save Stage: Actually saves a G Projection for the location code.

---

Note: Location Code

---

You will have to prepare the necessary folders and files to perform calibration, there is also a Location Tab to help you with creating the barebone location folder, ready for calibration. Below is an example structure:

---

```
TrafficLab-3D/
└── location/
    └── {location_code}/
        ├── footage/
        │   └── *.mp4
        │
        ├── illustrator/                 (optional, Adobe Illustrator assets)
        │   ├── layout_{location_code}.ai
        │   ├── roi_{location_code}.ai
        │   └── *.ai
        │
        ├── G_projection_{location_code}.json
        ├── cctv_{location_code}.png     (critical!)
        ├── sat_{location_code}.png      (critical!)
        ├── layout_{location_code}.svg   (optional)
        └── roi_{location_code}.png      (optional)
```

---

If you haven't noticed, you can create custom SVG and ROI using Adobe Illustrator, refer to the blog post/Youtube video for a more detailed guide on crafting said resources.

---

### 2.2. Inference Tab

---

Inference Tab is a Hub for you to keep track of all your production of the output JSON files, these files are what are actually used by the visualization engine, eliminating the need to perform demanding computation on top of heavy rendering. For arguments, you will control them through inference_config.yaml and prior_dimensions.json in the project's root. JSON will be saved as the compressed .json.gz format for storage efficiency. Controllable arguments includes:

- Object detection model.
- Object tracker.
- Speed and orientation smoothing kinematics.

---

### 2.3. Visualization Tab

---

The visualization engine for the output JSON files, features comprehensive controls via a tool bar and keyboard shortcuts, flexible side by side view of CCTV and SAT panel.

---

### Yuk's Remark

---

Thank you for checking out this program, at this 1.0 version, the code is pretty much just glued on research notes from my academic report, however, I really hope it can at least have it uses, maybe even for fun. It has been a really long time since I get to code a standalone application like this with GUI and all, I think last time had to be over a year ago where I still did everything with Java, back then I already had some interest in AI with the Gomoku (5-in-a-row) game with a bot that i coded up myself, it beat rusty players but was a 50/50 against other bots. Welp, enough reminiscing, simply put, I'm quite happy with what I was able to achieve with this program given the time and resource constraint.

---

Moreover, in the process of making TrafficLab 3D, I've also discovered my own enjoyment for this little niche-embodied and physically grounded AI, and plan to make more projects in this field rather than just vaguely "doing AI" like i've been doing before this program. Anyway, see ya around.

---

- Yuk - Dec. 25th, 2025
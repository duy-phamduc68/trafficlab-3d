```
TrafficLab-3D/
├── location/
│   └── {location_code}/
│       ├── footage/
│       │   └── *.mp4
│       │
│       ├── illustrator/                 (optional, Adobe Illustrator assets)
│       │   ├── layout_{location_code}.ai
│       │   ├── roi_{location_code}.ai
│       │   └── *.ai
│       │
│       ├── G_projection_{location_code}.json
│       ├── cctv_{location_code}.png     (critical!)
│       ├── sat_{location_code}.png      (critical!)
│       ├── layout_{location_code}.svg   (optional)
│       └── roi_{location_code}.png      (optional)
│
├── media/                               (resources for README and Introduction tab)
│
├── gui/                                 (GUI implementation)
│
├── models/                              (object detection & tracker models)
│   └── *.pt                             (YOLO checkpoints)
│
├── output/
│   └── model-{model_name}_tracker-{tracker_name}/
│       └── {config-name}/
│           └── {location_code}/
│               └── *.json.gz             (inference outputs)
│
├── environment.yml
├── inference_config.yaml
├── prior_dimensions.json
└── main.py
```
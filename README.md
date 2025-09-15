# Train Coach Analysis System

This project processes train videos to detect and analyze individual coaches, including door detection and coverage reporting.

## Features

- Video processing to detect and separate individual train coaches
- Coach counting and video segmentation
- Frame extraction for each coach
- Object detection for doors (open/closed)
- Coverage report generation
- Organized output directory structure

## Prerequisites

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- Other dependencies listed in requirements.txt

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Download the YOLOv8 weights:
   ```
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   ```

## Usage

1. Place your input video in the project directory
2. Run the main script:
   ```
   python train_coach_analyzer.py
   ```
3. The processed outputs will be saved in the `output` directory

## Project Structure

```
project/
├── output/
│   ├── coach_videos/     # Individual coach videos
│   ├── frames/           # Extracted frames
│   ├── detections/       # Frames with detections
│   └── reports/          # Generated reports
├── train_coach_analyzer.py  # Main processing script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Configuration

Edit the `train_coach_analyzer.py` file to adjust parameters like:
- Coach detection sensitivity
- Frame extraction rate
- Output directories
- Model parameters

## Output

The system generates:
- Individual coach videos
- Extracted frames for each coach
- Annotated frames with detections
- A comprehensive report in the `reports` directory

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

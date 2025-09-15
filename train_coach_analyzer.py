import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from datetime import datetime
import json
from fpdf import FPDF


class TrainCoachAnalyzer:
    def __init__(self, video_path, output_dir='output'):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.coach_videos_dir = self.output_dir / 'coach_videos'
        self.frames_dir = self.output_dir / 'frames'
        self.detections_dir = self.output_dir / 'detections'
        self.reports_dir = self.output_dir / 'reports'


        self.frame_skip = 10
        self.target_width = 640
        self.min_coach_area = 0.1
        self.min_coach_aspect = 0.2
        self.min_coach_frames = 5
        self.max_coach_gap = 150
        self.min_coach_width = 100
        self.min_coach_height = 50
        self.coach_separation_threshold = 50


        self._create_directories()


        self.model = YOLO('yolov8n.pt')


        self.cap = cv2.VideoCapture(str(video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))


    def _create_directories(self):
        self.coach_videos_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.detections_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


    def _resize_frame(self, frame, target_width=None):
        if target_width is None:
            target_width = self.target_width
        if frame is None:
            return None
        height, width = frame.shape[:2]
        if width == target_width:
            return frame
        aspect_ratio = height / width
        target_height = int(target_width * aspect_ratio)
        return cv2.resize(frame, (target_width, target_height))


    def _get_roi(self, frame, y_start=0.1, y_end=0.9, x_margin=0.05):
        if frame is None or frame.size == 0:
            return None
        height, width = frame.shape[:2]
        y1 = int(height * y_start)
        y2 = int(height * y_end)
        x1 = int(width * x_margin)
        x2 = int(width * (1 - x_margin))
        min_width_req = int(width * 0.7)
        min_height_req = int(height * 0.4)


        # Adjust width if needed
        if (x2 - x1) < min_width_req:
            center_x = (x1 + x2) // 2
            x1 = max(0, center_x - min_width_req // 2)
            x2 = min(width, center_x + min_width_req // 2)


        # Adjust height if needed
        if (y2 - y1) < min_height_req:
            center_y = (y1 + y2) // 2
            y1 = max(0, center_y - min_height_req // 2)
            y2 = min(height, center_y + min_height_req // 2)


        roi = frame[y1:y2, x1:x2]


        if roi.size == 0:
            print(f"Warning: empty ROI extracted (y:{y1}-{y2}, x:{x1}-{x2})")
            return None


        try:
            if roi.ndim == 3:
                lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                merged = cv2.merge((l, a, b))
                roi = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            else:
                roi = cv2.equalizeHist(roi)
        except Exception as e:
            print("CLAHE error:", e)


        return roi


    def _is_coach_visible(self, frame, min_contour_area=100):
        if frame is None or frame.size == 0:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_contour_area:
                return True
        return False


    def _get_bbox(self, frame):
        if frame is None or frame.size == 0:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5,5), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = 10
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(frame.shape[1] - x, w + 2 * pad)
        h = min(frame.shape[0] - y, h + 2 * pad)
        return (x, y, w, h)


    def process_video(self):
        print(f"Processing {self.video_path}")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        coach_boxes = []
        frame_num = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_num % self.frame_skip != 0:
                frame_num += 1
                continue
            if self._is_coach_visible(frame):
                bbox = self._get_bbox(frame)
                if bbox:
                    # Check if this bbox is far enough from existing ones
                    if all(not self._box_close(bbox, c['box']) for c in coach_boxes):
                        ctype = 'wagon'  # default type
                        if len(coach_boxes) == 0:
                            ctype = 'engine1'
                        # Removed engine2 from here
                        coach_boxes.append({'box': bbox, 'frame_num': frame_num, 'type': ctype})
                        print(f"Detected {ctype} at frame {frame_num} bbox {bbox}")
            frame_num += 1
        self.cap.release()


        # Organize and re-label coaches
        coach_boxes = self._organize_coaches(coach_boxes)
        return coach_boxes


    def _box_close(self, b1, b2, threshold=100):
        # Simple distance between box centers to check close detections
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        c1 = (x1 + w1 // 2, y1 + h1 // 2)
        c2 = (x2 + w2 // 2, y2 + h2 // 2)
        dist = np.linalg.norm(np.array(c1) - np.array(c2))
        return dist < threshold


    def _organize_coaches(self, coach_boxes):
        # Sort by frame_num and assign proper numbering
        coach_boxes.sort(key=lambda x: x['frame_num'])
        # Separate wagons and engines
        wagons = [c for c in coach_boxes if 'engine' not in c['type']]
        engines = [c for c in coach_boxes if 'engine' in c['type']]

        # Assign numbered wagons except last one
        for idx, c in enumerate(wagons[:-1], start=1):
            c['type'] = f'wagon{idx}'

        # Rename last wagon as tailcoach if wagons exist
        if wagons:
            wagons[-1]['type'] = 'tailcoach'

        # Combine list: engines first, wagons after
        return engines + wagons


    def extract_coach_videos(self, coach_boxes):
        if not coach_boxes:
            print("No coaches detected, skipping extraction.")
            return
        print(f"Extracting videos for {len(coach_boxes)} coaches")
        cap = cv2.VideoCapture(str(self.video_path))
        for idx, c in enumerate(coach_boxes, start=1):
            folder = self.coach_videos_dir / f"coach_{idx:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            bbox = c['box']
            start_frame = max(0, c['frame_num'] - int(self.fps))  # 1 sec before detected
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            video_path = folder / f"coach_{idx:02d}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, self.fps, (bbox[2], bbox[3]))
            frames_written = 0
            while frames_written < int(2 * self.fps):  # 2 sec video clip
                ret, frame = cap.read()
                if not ret:
                    break
                subframe = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                if subframe.size > 0:
                    out.write(subframe)
                    frames_written += 1
            out.release()
            # Save first frame image
            cap.set(cv2.CAP_PROP_POS_FRAMES, c['frame_num'])
            ret, frame = cap.read()
            if ret:
                subframe = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                cv2.imwrite(str(folder / "first_frame.jpg"), subframe)
        cap.release()
        print("Extraction complete.")


    def process_coach_frames(self, coach, coach_dir):
        folder = self.coach_videos_dir / coach_dir.name
        frames_folder = self.frames_dir / coach_dir.name
        detections_folder = self.detections_dir / coach_dir.name
        frames_folder.mkdir(parents=True, exist_ok=True)
        detections_folder.mkdir(parents=True, exist_ok=True)
        video_path = coach_dir / f"{coach_dir.name}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Cannot open coach video {video_path}")
            return
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in tqdm(range(frame_count), desc=f'Processing frames for {coach_dir.name}'):
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = frames_folder / f"frame_{i:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            results = self.model(frame)
            if results and hasattr(results[0], 'plot'):
                annotated = results[0].plot()
                det_path = detections_folder / f"detection_{i:04d}.jpg"
                cv2.imwrite(str(det_path), annotated)
        cap.release()


    def generate_report(self, coach_boxes):
        coach_boxes = self._organize_coaches(coach_boxes)
        report_dir = self.reports_dir
        images_dir = report_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        pdf = FPDF()
        pdf.set_title("Train Coach Analysis Report")
        pdf.set_author("TrainCoachAnalyzer")
        pdf.set_auto_page_break(auto=True, margin=15)


        # Cover page
        pdf.add_page()
        pdf.set_font("Arial", 'B', 24)
        pdf.set_text_color(0, 60, 113)
        pdf.cell(0, 40, "TRAIN COACH ANALYSIS REPORT", align='C', ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, f"Video: {self.video_path.name}", align='C', ln=True)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align='C', ln=True)
        pdf.ln(10)


        # Summary
        counts = {}
        for c in coach_boxes:
            t = c['type']
            counts[t] = counts.get(t, 0) + 1
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Summary of Detected Coaches:", ln=True)
        pdf.set_font("Arial", '', 14)
        for k, v in counts.items():
            pdf.cell(0, 10, f"{k.capitalize()} : {v}", ln=True)
        pdf.cell(0, 10, f"Total coaches detected: {len(coach_boxes)}", ln=True)
        pdf.add_page()


        # Add each coach details
        for idx, c in enumerate(coach_boxes, start=1):
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, f"{c['type'].capitalize()} #{idx}", ln=True)
            bbox = c['box']
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 10, f"First detected in frame: {c['frame_num']}", ln=True)
            pdf.cell(0, 10, f"Bbox position: {bbox}", ln=True)
            coach_folder = self.coach_videos_dir / f"coach_{idx:02d}"
            first_frame_path = coach_folder / "first_frame.jpg"
            if first_frame_path.exists():
                pdf.image(str(first_frame_path), w=180)
            pdf.add_page()


        output_path = self.reports_dir / "train_coaches_report.pdf"
        pdf.output(str(output_path))
        print(f"Report saved: {output_path}")
        return str(output_path)


def main():
    video_file = "train view.mp4"
    if not os.path.exists(video_file):
        print(f"Video file '{video_file}' not found. Please place the file in the current directory.")
        return


    analyzer = TrainCoachAnalyzer(video_file)
    try:
        print("Starting processing...")
        coach_boxes = analyzer.process_video()


        print("Extracting coach videos...")
        analyzer.extract_coach_videos(coach_boxes)


        for idx, coach in enumerate(coach_boxes, start=1):
            coach_dir = analyzer.coach_videos_dir / f"coach_{idx:02d}"
            coach_dir.mkdir(parents=True, exist_ok=True)
            print(f"Processing frames for coach #{idx}...")
            analyzer.process_coach_frames(coach, coach_dir)


        print("Generating final report...")
        analyzer.generate_report(coach_boxes)


        print("Processing completed. Check outputs in 'output' directory.")


    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

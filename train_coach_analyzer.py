import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from datetime import datetime
import json

class TrainCoachAnalyzer:
    def __init__(self, video_path, output_dir='output'):
        """
        Initialize the TrainCoachAnalyzer with optimization settings
        
        Args:
            video_path (str): Path to the input video file
            output_dir (str): Directory to save all outputs
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.coach_videos_dir = self.output_dir / 'coach_videos'
        self.frames_dir = self.output_dir / 'frames'
        self.detections_dir = self.output_dir / 'detections'
        self.reports_dir = self.output_dir / 'reports'
        
        # Optimization parameters
        self.frame_skip = 10  # Process every 10th frame for better detection
        self.target_width = 640  # Target width for resizing frames
        self.min_coach_area = 0.1  # Reduced minimum area to detect smaller coaches
        self.min_coach_aspect = 0.2  # More permissive aspect ratio
        self.min_coach_frames = 5  # Reduced minimum frames for detection
        self.max_coach_gap = 150  # Increased gap between coaches
        self.min_coach_width = 100  # Minimum width in pixels
        self.min_coach_height = 50  # Minimum height in pixels
        
        # Create necessary directories
        self._create_directories()
        
        # Initialize YOLO model for door detection with optimizations
        self.model = YOLO('yolov8n.pt')  # Using YOLOv8 nano for faster inference
        
        # Video properties
        self.cap = cv2.VideoCapture(str(video_path))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Coach detection parameters
        self.coach_separation_threshold = 50  # Pixels between coaches
        self.min_coach_width = 100  # Minimum width of a coach in pixels
        
    def _create_directories(self):
        """Create necessary output directories"""
        self.coach_videos_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.detections_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def _resize_frame(self, frame, target_width=None):
        """Resize frame to target width while maintaining aspect ratio"""
        if target_width is None:
            target_width = self.target_width
            
        if frame is None:
            return None
            
        height, width = frame.shape[:2]
        if width == target_width:
            return frame
            
        # Calculate the target height to maintain aspect ratio
        aspect_ratio = height / width
        target_height = int(target_width * aspect_ratio)
        
        # Resize the frame
        return cv2.resize(frame, (target_width, target_height))
    
    def _get_roi(self, frame, y_start=0.1, y_end=0.9, x_margin=0.05):
        """Get region of interest for coach detection with improved boundary handling
        
        Args:
            frame: Input frame
            y_start: Starting y-coordinate as fraction of frame height (0-1)
            y_end: Ending y-coordinate as fraction of frame height (0-1)
            x_margin: Horizontal margin ratio (0-0.5) from both sides
            
        Returns:
            ROI image or None if invalid
        """
        if frame is None or frame.size == 0:
            return None
            
        height, width = frame.shape[:2]
        
        # Calculate ROI coordinates - use most of the frame width and height
        y1 = int(height * y_start)
        y2 = int(height * y_end)
        x1 = int(width * x_margin)  # Use provided x_margin for horizontal padding
        x2 = int(width * (1 - x_margin))
        
        # Ensure we have a wide enough capture area
        min_width = int(width * 0.8)  # At least 80% of frame width
        current_width = x2 - x1
        if current_width < min_width:
            # Center the ROI and expand to minimum width
            center_x = (x1 + x2) // 2
            x1 = max(0, center_x - min_width // 2)
            x2 = min(width, center_x + min_width // 2)
        
        # Ensure coordinates are within frame bounds
        y1 = max(0, y1)
        y2 = min(height, y2)
        x1 = max(0, x1)
        x2 = min(width, x2)
        
        # Ensure minimum dimensions for better detection
        min_height = int(height * 0.4)  # Increased minimum height to 40%
        min_width = int(width * 0.7)    # Increased minimum width to 70%
        
        # Calculate current dimensions
        current_height = y2 - y1
        current_width = x2 - x1
        
        # Adjust height if needed
        if current_height < min_height:
            center_y = (y1 + y2) // 2
            new_half_height = min_height // 2
            y1 = max(0, center_y - new_half_height)
            y2 = min(height, center_y + new_half_height)
            
            # If we hit the top or bottom, try to expand the other way
            if y1 == 0 and y2 < height:
                y2 = min(height, y2 + (min_height - (y2 - y1)))
            elif y2 == height and y1 > 0:
                y1 = max(0, y1 - (min_height - (y2 - y1)))
        
        # Adjust width if needed
        if current_width < min_width:
            center_x = (x1 + x2) // 2
            new_half_width = min_width // 2
            x1 = max(0, center_x - new_half_width)
            x2 = min(width, center_x + new_half_width)
            
            # If we hit the left or right, try to expand the other way
            if x1 == 0 and x2 < width:
                x2 = min(width, x2 + (min_width - (x2 - x1)))
            elif x2 == width and x1 > 0:
                x1 = max(0, x1 - (min_width - (x2 - x1)))
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            print(f"Warning: Empty ROI extracted - y:{y1}-{y2}, x:{x1}-{x2}")
            return None
            
        # Apply CLAHE for better contrast
        try:
            if len(roi.shape) == 3:  # Color image
                lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl,a,b))
                roi = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            else:  # Grayscale
                roi = cv2.equalizeHist(roi)
                
        except Exception as e:
            print(f"Error in CLAHE: {str(e)}")
            
        return roi
    
    def _is_coach_visible(self, frame, min_contour_area=100):
        """Enhanced coach visibility check using multiple detection methods"""
        print("\n--- Starting coach visibility check ---")
        
        if frame is None or frame.size == 0:
            print("Error: Empty frame received")
            return False
        
        debug_dir = self.output_dir / 'debug_frames'
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            cv2.imwrite(str(debug_dir / '00_original.jpg'), frame)
            print("Saved original frame")
                
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
            
            cv2.imwrite(str(debug_dir / '01_grayscale.jpg'), gray)
            print("Converted to grayscale")
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_img = clahe.apply(gray)
            cv2.imwrite(str(debug_dir / '02_clahe.jpg'), clahe_img)
            
            thresh = cv2.adaptiveThreshold(clahe_img, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            cv2.imwrite(str(debug_dir / '03_adaptive_thresh.jpg'), thresh)
            
            kernel = np.ones((3,3), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
            cv2.imwrite(str(debug_dir / '04_cleaned.jpg'), cleaned)
            
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"Found {len(contours)} total contours")
            
            contour_img = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
            
            valid_contours = []
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area < 10:
                    continue
                    
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w)/h if h > 0 else 0
                
                cv2.rectangle(contour_img, (x, y), (x+w, y+h), (255, 0, 0), 1)
                
                min_width = 30
                min_height = 30
                is_valid = (area > min_contour_area and 
                          0.05 < aspect_ratio < 20.0 and
                          w > min_width and 
                          h > min_height and
                          w < frame.shape[1] * 0.9 and
                          h < frame.shape[0] * 0.9)
                
                if is_valid:
                    valid_contours.append((cnt, area, aspect_ratio, w, h))
            
            print(f"Found {len(valid_contours)} potential coach contours")
            
            valid_contours.sort(key=lambda x: x[1], reverse=True)
            
            top_contours = valid_contours[:3]
            
            for cnt_data in top_contours:
                cnt = cnt_data[0]
                area = cnt_data[1]
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                print(f"  - Potential coach: area={area}, aspect={cnt_data[2]:.2f}, size=({w}x{h})")
            
            cv2.imwrite(str(debug_dir / '05_contours.jpg'), contour_img)
            
            return len(top_contours) > 0
            
        except Exception as e:
            print(f"Error in _is_coach_visible: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_coach_bbox(self, frame, min_confidence=0.01, max_aspect_ratio=2.5):
        """Enhanced coach bounding box detection with multiple validation steps
        
        Args:
            frame: Input frame to detect coach in
            min_confidence: Minimum confidence threshold for detection
            max_aspect_ratio: Maximum allowed width/height ratio for the bounding box
            
        Returns:
            Tuple of (x, y, w, h) or None if no valid detection
        """
        print("\n--- Getting coach bounding box ---")
        if frame is None or frame.size == 0:
            print("Error: Empty frame in _get_coach_bbox")
            return None
            
        orig_h, orig_w = frame.shape[:2]
        debug_dir = self.output_dir / 'debug_frames'
        debug_dir.mkdir(parents=True, exist_ok=True)
            
        roi = self._get_roi(frame)
        if roi is None:
            return None
            
        debug_img = roi.copy()
        
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equalized = clahe.apply(gray)
        
        filtered = cv2.bilateralFilter(equalized, 9, 75, 75)
        
        edges = cv2.Canny(filtered, 30, 150)
        
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_score = 0
        best_box = None
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            if w < self.min_coach_width or h < self.min_coach_height:
                continue
                
            aspect_ratio = float(w) / h
            
            area = cv2.contourArea(cnt)
            rect_area = w * h
            extent = float(area) / rect_area if rect_area > 0 else 0
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            score = (extent * 0.4 + 
                    solidity * 0.3 + 
                    (1 - abs(1 - aspect_ratio)) * 0.3)
            
            if score > best_score and score > min_confidence:
                best_score = score
                best_box = (x, y, w, h)
        
        if best_box is None:
            return None
            
        x, y, w, h = best_box
        
        is_tail_coach = (y + h/2) > (frame.shape[0] * 0.5)
        
        if is_tail_coach:
            print("  Tail coach detected - applying expanded padding")
            padding_x = int(w * 0.3)
            padding_y_top = int(h * 0.2)
            padding_y_bottom = int(h * 0.4)
            
            x = max(0, x - padding_x)
            y = max(0, y - padding_y_top)
            w = min(roi.shape[1] - x, w + 2 * padding_x)
            h = min(roi.shape[0] - y, h + padding_y_top + padding_y_bottom)
            
            min_width = int(frame.shape[1] * 0.4)
            h_min = int(h * 1.2)
            w = max(w, min_width)
            h = max(h, h_min)
        else:
            padding = int(min(w, h) * 0.15)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(roi.shape[1] - x, w + 2 * padding)
            h = min(roi.shape[0] - y, h + 2 * padding)
        
        roi_height, roi_width = roi.shape[:2]
        frame_height, frame_width = frame.shape[:2]
        
        x_scale = frame_width / roi_width
        y_scale = frame_height / roi_height
        
        x = int(x * x_scale)
        y = int(y * y_scale)
        w = int(w * x_scale)
        h = int(h * y_scale)
        
        x = max(0, min(x, frame_width - 1))
        y = max(0, min(y, frame_height - 1))
        w = max(1, min(w, frame_width - x))
        h = max(1, min(h, frame_height - y))
        
        if w < self.min_coach_width or h < self.min_coach_height:
            print(f"  Box too small: {w}x{h} (min {self.min_coach_width}x{self.min_coach_height})")
            return None
        
        min_coach_width = int(frame.shape[1] * 0.3)
        if w < min_coach_width and is_tail_coach:
            center_x = x + w/2
            x = max(0, int(center_x - min_coach_width/2))
            w = min(frame.shape[1] - x, min_coach_width)
            print(f"  Adjusted width for tail coach: {w}x{h}")
            
        center_x = x + w/2
        frame_center = frame.shape[1] / 2
        horizontal_offset = abs(center_x - frame_center) / frame_center
        
        print(f"  Final bbox: x={x}, y={y}, w={w}, h={h}, is_tail={is_tail_coach}")
        return (x, y, w, h)
    
    def _is_too_close(self, bbox1, bbox2):
        """Check if two bounding boxes are too close based on horizontal distance"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        center1 = x1 + w1 // 2
        center2 = x2 + w2 // 2
        return abs(center1 - center2) < self.coach_separation_threshold
    
    def process_video(self):
        """Process the video to detect coach types (head, middle, tail)"""
        print(f"Processing video: {self.video_path}")
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError("Could not open the video file")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Total frames: {total_frames} (processing every {self.frame_skip} frames)")
        print("Detecting coaches...")
        
        coach_boxes = []
        
        def add_coach_if_valid(bbox, frame_num, coach_type):
            if not bbox:
                return False
                
            x, y, w, h = bbox
            min_width = max(50, self.min_coach_width // 2)
            min_height = max(30, self.min_coach_height // 2)
            if w < min_width or h < min_height or w > self.frame_width * 0.9 or h > self.frame_height * 0.9:
                return False
                
            for coach in coach_boxes:
                if self._is_too_close(bbox, coach['box']):
                    return False
            
            coach_boxes.append({
                'type': coach_type,
                'box': bbox,
                'frame_num': frame_num
            })
            print(f"Detected {coach_type} coach at frame {frame_num} with bbox {bbox}")
            return True
        
        print("Detecting head coach...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            head_bbox = self._get_coach_bbox(frame)
            add_coach_if_valid(head_bbox, 0, 'head')
        
        print("Sampling sections for middle coaches...")
        sample_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        for pos_ratio in sample_points:
            pos = int(total_frames * pos_ratio)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Failed to read frame at position {pos}")
                continue
                
            print(f"\n--- Processing frame at {pos_ratio*100:.0f}% of video (frame {pos}) ---")
            
            debug_dir = self.output_dir / 'debug_frames'
            debug_dir.mkdir(exist_ok=True, parents=True)
            
            timestamp = f"{pos:05d}_{int(pos_ratio*100):03d}pc"
            cv2.imwrite(str(debug_dir / f'frame_{timestamp}_original.jpg'), frame)
            
            print("Trying full frame detection...")
            if self._is_coach_visible(frame):
                full_bbox = self._get_coach_bbox(frame)
                if full_bbox and add_coach_if_valid(full_bbox, pos, 'middle'):
                    print(f"✅ Found coach using full frame at position {pos_ratio:.2f}")
                    x, y, w, h = full_bbox
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.imwrite(str(debug_dir / f'frame_{timestamp}_detected.jpg'), frame)
                    continue
            
            print("Trying different ROIs...")
            found = False
            for y_start in [0.1, 0.3, 0.5, 0.7]:
                for y_height in [0.3, 0.5]:
                    roi = self._get_roi(frame, y_start=y_start, y_end=y_start+y_height)
                    if roi is None:
                        print(f"  ❌ Failed to get ROI y={y_start:.1f}, h={y_height}")
                        continue
                        
                    roi_debug_dir = debug_dir / f'roi_y{y_start:.1f}_h{y_height:.1f}'
                    roi_debug_dir.mkdir(exist_ok=True, parents=True)
                    
                    roi_filename = f'roi_{timestamp}.jpg'
                    cv2.imwrite(str(roi_debug_dir / roi_filename), roi)
                    
                    print(f"  🔍 Checking ROI: y={y_start:.1f}, height={y_height}")
                    if not self._is_coach_visible(roi):
                        print(f"    ❌ No coach visible in ROI")
                        continue
                    
                    middle_bbox = self._get_coach_bbox(roi)
                    
                    if middle_bbox:
                        print(f"    ✅ Found potential coach in ROI")
                        frame_h, frame_w = frame.shape[:2]
                        roi_h, roi_w = roi.shape[:2]
                        
                        scale_x = frame_w / roi_w
                        scale_y = frame_h / roi_h
                        
                        x, y, w, h = middle_bbox
                        frame_x = int(x * scale_x)
                        frame_y = int(y * scale_y + y_start * frame_h)
                        frame_w = int(w * scale_x)
                        frame_h = int(h * scale_y)
                        
                        frame_bbox = (frame_x, frame_y, frame_w, frame_h)
                        
                        if add_coach_if_valid(frame_bbox, pos, 'middle'):
                            print(f"🎯 Found middle coach at position {pos_ratio:.2f} with ROI y={y_start}, h={y_height}")
                            cv2.rectangle(frame, (frame_x, frame_y), 
                                        (frame_x + frame_w, frame_y + frame_h), 
                                        (0, 255, 0), 3)
                            cv2.imwrite(str(debug_dir / f'frame_{timestamp}_detected_roi.jpg'), frame)
                            found = True
                            break
                if found:
                    break
        
        print("\n=== LOOKING FOR TAIL COACH ===")
        tail_start = int(total_frames * 0.7)
        tail_end = total_frames - 1
        tail_step = max(1, (tail_end - tail_start) // 30)
        
        potential_tails = []
        
        for pos in range(tail_start, tail_end, tail_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            
            if not ret:
                print(f"  ❌ Failed to read frame {pos}")
                continue
                
            print(f"\n--- Checking for tail at frame {pos} ({(pos/total_frames*100):.1f}%) ---")
            
            debug_dir = self.output_dir / 'debug_frames'
            debug_dir.mkdir(exist_ok=True, parents=True)
            timestamp = f"tail_{pos:05d}"
            cv2.imwrite(str(debug_dir / f'{timestamp}_original.jpg'), frame)
            
            print("  Trying full frame detection...")
            if self._is_coach_visible(frame):
                tail_bbox = self._get_coach_bbox(frame, min_confidence=0.005)
                if tail_bbox:
                    x, y, w, h = tail_bbox
                    print(f"  ✅ Found potential tail with full frame: {tail_bbox}")
                    potential_tails.append((tail_bbox, pos, 'full_frame'))
                    
                    debug_frame = frame.copy()
                    cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(debug_frame, 'Tail?', (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imwrite(str(debug_dir / f'{timestamp}_full_frame_detection.jpg'), debug_frame)
            
            print("  Trying different ROIs...")
            for y_start in [0.1, 0.3, 0.5, 0.7]:
                for y_height in [0.3, 0.4, 0.5]:
                    roi = self._get_roi(frame, y_start=y_start, y_end=y_start+y_height, x_margin=0.05)
                    if roi is None:
                        continue
                        
                    roi_debug_dir = debug_dir / 'tail_rois'
                    roi_debug_dir.mkdir(exist_ok=True, parents=True)
                    roi_filename = f'{timestamp}_y{y_start:.1f}_h{y_height:.1f}.jpg'
                    cv2.imwrite(str(roi_debug_dir / roi_filename), roi)
                    
                    if self._is_coach_visible(roi):
                        tail_bbox = self._get_coach_bbox(roi, min_confidence=0.005)
                        
                        if tail_bbox:
                            frame_h, frame_w = frame.shape[:2]
                            roi_h, roi_w = roi.shape[:2]
                            
                            scale_x = frame_w / roi_w
                            scale_y = frame_h / roi_h
                            
                            x, y, w, h = tail_bbox
                            frame_x = int(x * scale_x)
                            frame_y = int(y * scale_y + y_start * frame_h)
                            frame_w = int(w * scale_x)
                            frame_h = int(h * scale_y)
                            
                            frame_bbox = (frame_x, frame_y, frame_w, frame_h)
                            print(f"    ✅ Found potential tail in ROI y={y_start:.1f}, h={y_height}: {frame_bbox}")
                            potential_tails.append((frame_bbox, pos, f'roi_y{y_start:.1f}'))
                            
                            debug_frame = frame.copy()
                            cv2.rectangle(debug_frame, (frame_x, frame_y), 
                                        (frame_x + frame_w, frame_y + frame_h), 
                                        (0, 255, 255), 2)
                            cv2.putText(debug_frame, f'Tail? y={y_start:.1f}', 
                                       (frame_x, frame_y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            cv2.imwrite(str(roi_debug_dir / f'detected_{roi_filename}'), debug_frame)
        
        if potential_tails:
            print("\n=== PROCESSING POTENTIAL TAIL COACHES ===")
            potential_tails.sort(key=lambda x: x[0][2] * x[0][3], reverse=True)
            
            for i, (tail_bbox, pos, detection_type) in enumerate(potential_tails[:3]):
                print(f"  Candidate {i+1} (from {detection_type} at frame {pos}): {tail_bbox}")
                
                if i == 0 and add_coach_if_valid(tail_bbox, pos, 'tail'):
                    print(f"🎯 Selected best tail coach at frame {pos}: {tail_bbox}")
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                    ret, frame = cap.read()
                    if ret:
                        x, y, w, h = tail_bbox
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                        cv2.putText(frame, 'TAIL COACH', (x, y-20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imwrite(str(debug_dir / 'selected_tail_coach.jpg'), frame)
                    break
        
        cap.release()
        
        coach_boxes.sort(key=lambda x: x['frame_num'])
        
        for i, coach in enumerate(coach_boxes):
            print(f"\nProcessing coach {i+1} (Type: {coach.get('type', 'unknown')})...")
            coach_dir = self.coach_videos_dir / f'coach_{i+1:03d}'
            coach_dir.mkdir(parents=True, exist_ok=True)
            
            cap = cv2.VideoCapture(str(self.video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, coach['frame_num'])
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(str(coach_dir / 'first_frame.jpg'), frame)
            cap.release()
            
            self.process_coach_frames(coach, coach_dir, save_sample_frames=5)
        
        print(f"\nDetected {len(coach_boxes)} coach types in the video")
        return coach_boxes
    
    def extract_coach_videos(self, coach_boxes):
        if not coach_boxes:
            print("No coach boxes provided for video extraction")
            return
            
        self.coach_videos_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError("Could not open the video file")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Found {len(coach_boxes)} coaches to extract")
        
        for i, coach in enumerate(coach_boxes, 1):
            frame_num = coach.get('frame_num', 0)
            x, y, w, h = coach['box']
            
            print(f"Extracting coach {i} from frame {frame_num}...")
            
            coach_dir = self.coach_videos_dir / f"coach_{i:03d}"
            coach_dir.mkdir(exist_ok=True)
            
            output_path = coach_dir / f"coach_{i:03d}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            
            start_frame = max(0, frame_num - int(fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames_to_capture = int(fps * 2)
            frames_captured = 0
            
            while frames_captured < frames_to_capture and start_frame + frames_captured < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    out.write(roi)
                    frames_captured += 1
                else:
                    print(f"Warning: Invalid ROI for coach {i} at frame {start_frame + frames_captured}")
                    
            out.release()
            print(f"Saved {frames_captured} frames for coach {i} to {output_path}")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    first_frame_path = coach_dir / "first_frame.jpg"
                    cv2.imwrite(str(first_frame_path), roi)
                    
        cap.release()
        print("Finished extracting all coach videos")
    
    def process_coach_frames(self, coach, coach_dir, save_sample_frames=5):
        coach_id = coach_dir.name
        frames_output_dir = self.frames_dir / coach_id
        detections_output_dir = self.detections_dir / coach_id
        frames_output_dir.mkdir(parents=True, exist_ok=True)
        detections_output_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = coach_dir / f"{coach_id}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            print(f"Warning: No frames in video {video_path}")
            cap.release()
            return
            
        sample_indices = set(np.linspace(0, frame_count-1, min(save_sample_frames, frame_count), dtype=int))
        
        for frame_idx in tqdm(range(frame_count), desc=f"Processing {coach_id}"):
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx in sample_indices:
                frame_path = frames_output_dir / f"frame_{frame_idx:04d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                
                results = self.model(frame)
                
                if hasattr(results, 'render') and len(results) > 0:
                    rendered = results[0].plot()
                    det_path = detections_output_dir / f"detection_{frame_idx:04d}.jpg"
                    cv2.imwrite(str(det_path), rendered)
        
        cap.release()
        print(f"Finished processing {coach_id}")
    
    def generate_report(self, coach_boxes):
        from fpdf import FPDF
        
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Train Coach Analysis Report', 0, 1, 'C')
        pdf.ln(10)
        
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Video: {self.video_path}', 0, 1)
        pdf.cell(0, 10, f'Analysis time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
        pdf.ln(5)
        
        if not coach_boxes:
            pdf.cell(0, 10, 'No coaches were detected in the video.', 0, 1)
            pdf_path = self.reports_dir / 'coach_analysis_report.pdf'
            pdf.output(str(pdf_path))
            return str(pdf_path)
        
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, f'Detected {len(coach_boxes)} coach types:', 0, 1)
        pdf.ln(5)
        
        for i, coach in enumerate(coach_boxes, 1):
            coach_type = coach.get('type', 'unknown')
            frame_num = coach.get('frame_num', 0)
            
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, f'Coach {i} (Type: {coach_type}):', 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 8, f'- Detected at frame: {frame_num}', 0, 1)
            pdf.cell(0, 8, f'- Bounding box: {coach.get("box", "N/A")}', 0, 1)
            coach_dir = self.coach_videos_dir / f'coach_{i:03d}'
            first_frame = coach_dir / 'first_frame.jpg'
            
            if first_frame.exists():
                try:
                    pdf.image(str(first_frame), x=10, w=180)
                except Exception as e:
                    pdf.cell(0, 8, f'- Could not load image: {str(e)}', 0, 1)
            
            pdf.ln(5)
        
        pdf_path = self.reports_dir / 'coach_analysis_report.pdf'
        pdf.output(str(pdf_path))
        
        txt_path = self.reports_dir / 'analysis_report.txt'
        with open(txt_path, 'w') as f:
            f.write("=== Train Coach Analysis Report ===\n\n")
            f.write(f"Video: {self.video_path}\n")
            f.write(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Detected {len(coach_boxes)} coach types:\n")
            for i, coach in enumerate(coach_boxes, 1):
                f.write(f"\nCoach {i} (Type: {coach.get('type', 'unknown')}):\n")
                f.write(f"- Frame: {coach.get('frame_num', 'N/A')}\n")
                f.write(f"- Bounding box: {coach.get('box', 'N/A')}\n")
        
        print(f"\nPDF Report generated at: {pdf_path}")
        print(f"Text Report generated at: {txt_path}")
        return str(pdf_path)

def main():
    input_video = "train view.mp4"
    
    if not os.path.exists(input_video):
        print(f"Error: Video file '{input_video}' not found in the current directory.")
        print("Please make sure the video file is in the same directory as this script.")
        return
    
    print(f"Starting analysis of video: {input_video}")
    analyzer = TrainCoachAnalyzer(input_video)
    
    try:
        print("Step 1/4: Processing video to detect coaches...")
        coach_boxes = analyzer.process_video()
        
        print("\nStep 2/4: Extracting individual coach videos...")
        analyzer.extract_coach_videos(coach_boxes)
        
        print("\nStep 3/4: Processing frames for door detection...")
        for i, coach in enumerate(coach_boxes, 1):
            coach_dir = analyzer.coach_videos_dir / f'coach_{i:03d}'
            coach_dir.mkdir(parents=True, exist_ok=True)
            print(f"Processing frames for coach_{i:03d}...")
            analyzer.process_coach_frames(coach, coach_dir)
        
        print("\nStep 4/4: Generating final report...")
        analyzer.generate_report(coach_boxes)
        
        print("\nProcessing complete! Check the 'output' directory for results.")
    except Exception as e:
        print(f"\nAn error occurred during processing: {str(e)}")
        print("Please check the video file and try again.")

if __name__ == "__main__":
    main()

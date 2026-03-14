import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import torch
import pyttsx3
import os

model = YOLO("yolov8n.pt")

# List of videos to process
VIDEO_FILES = ["traffic.mp4"]

# Initialize text-to-speech engine ONCE
engine = pyttsx3.init()
engine.setProperty('rate', 125)  # Slower speech
engine.setProperty('volume', 1.0)  # MAXIMUM volume

def speak_text(text):
    """Speak text using pyttsx3"""
    try:
        engine.say(text)
        engine.runAndWait()
        print(f"🔊 Audio: {text}")
    except Exception as e:
        print(f"❌ Audio error: {e}")

# Constants
DIRECTION_THRESHOLD = 15
SIGNAL_CYCLE_TIME = 30  # Total cycle time in seconds (15s green + 15s red)
RED_LIGHT_DURATION = 15  # Duration of red light in seconds
GREEN_LIGHT_DURATION = 15  # Duration of green light in seconds

def get_signal_state(signal_start_time):
    """Returns current signal state: 'RED' or 'GREEN'"""
    elapsed = time.time() - signal_start_time
    cycle_position = elapsed % SIGNAL_CYCLE_TIME
    
    if cycle_position < GREEN_LIGHT_DURATION:
        return "GREEN"
    else:
        return "RED"

def is_vehicle_in_signal_zone(cy, SIGNAL_ZONE_Y_START, SIGNAL_ZONE_Y_END):
    """Check if vehicle center Y is in signal zone"""
    return SIGNAL_ZONE_Y_START <= cy <= SIGNAL_ZONE_Y_END

def process_video(video_file):
    """Process a single video file"""
    
    print(f"\n{'='*50}")
    print(f"Processing: {video_file}")
    print(f"{'='*50}\n")
    
    # Check if file exists
    if not os.path.exists(video_file):
        print(f"❌ Video file not found: {video_file}")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_file)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Generate output filename
    video_name = os.path.splitext(video_file)[0]
    output_file = f"output_{video_name}.mp4"
    
    out = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )
    
    # Reset tracking variables for this video
    track_history = {}
    vehicles_counted = set()
    vehicle_count = 0
    signal_jumping_count = 0
    wrong_way_count = 0
    vehicles_in_red = set()
    announced_violations = {}
    
    # Reset signal timing
    signal_start_time = time.time()
    COUNTING_LINE = int(height // 2)
    SIGNAL_ZONE_Y_START = int(height // 2) - 50
    SIGNAL_ZONE_Y_END = int(height // 2) + 50
    
    # Create tracker for this video
    tracker = DeepSort(max_age=40)
    
    # Process video frames
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Variables to track violations this frame
        frame_has_wrong_way = False
        frame_has_signal_jump = False
        
        results = model(frame, stream=True)
        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls in [2, 3, 5, 7] and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, w, h = map(int, track.to_ltrb())
            cx = l + w // 2
            cy = t + h // 2

            if track_id not in track_history:
                track_history[track_id] = []

            track_history[track_id].append(cy)
            if len(track_history[track_id]) > 10:
                track_history[track_id].pop(0)

            # COUNT VEHICLES CROSSING THE LINE
            if len(track_history[track_id]) >= 2:
                prev_cy = track_history[track_id][-2]
                curr_cy = track_history[track_id][-1]
                
                # Check if vehicle crossed the counting line
                if (prev_cy < COUNTING_LINE and curr_cy >= COUNTING_LINE) or \
                   (prev_cy > COUNTING_LINE and curr_cy <= COUNTING_LINE):
                    if track_id not in vehicles_counted: 
                        vehicles_counted.add(track_id)
                        vehicle_count += 1
                        print(f"Vehicle counted! Total: {vehicle_count}")

            # ===== SIGNAL JUMPING DETECTION =====
            current_signal = get_signal_state(signal_start_time)
            
            # Check if vehicle is in signal zone
            if is_vehicle_in_signal_zone(cy, SIGNAL_ZONE_Y_START, SIGNAL_ZONE_Y_END):
                # If vehicle entered signal zone during RED light
                if current_signal == "RED" and track_id not in vehicles_in_red:
                    vehicles_in_red.add(track_id)
                    signal_jumping_count += 1
                    print(f"⚠️  SIGNAL JUMPED! Vehicle ID {track_id} crossed RED light. Total violations: {signal_jumping_count}")
            else:
                # Vehicle left signal zone, remove from red light tracking
                if track_id in vehicles_in_red:
                    vehicles_in_red.discard(track_id)

            color = (0, 255, 0)
            label = f"ID {track_id}"
            
            # Check for WRONG WAY DETECTION
            has_wrong_way = False
            if len(track_history[track_id]) >= 2:
                dy = track_history[track_id][-1] - track_history[track_id][0]
                if dy < -DIRECTION_THRESHOLD:
                    has_wrong_way = True
                    frame_has_wrong_way = True  # Flag for this frame
                    if track_id not in announced_violations or not announced_violations[track_id].get('wrong_way'):
                        wrong_way_count += 1
                        announced_violations[track_id] = announced_violations.get(track_id, {})
                        announced_violations[track_id]['wrong_way'] = True
                    print(f"🚨 Wrong-way driving: Vehicle ID {track_id}")
            
            # Check for SIGNAL JUMPING
            has_signal_jump = False
            if track_id in vehicles_in_red:
                has_signal_jump = True
                frame_has_signal_jump = True  # Flag for this frame
                print(f"🟠 Signal jump: Vehicle ID {track_id}")
            
            # Determine box color based on violations
            if has_wrong_way and has_signal_jump:
                # Both violations = Purple box
                color = (255, 0, 255)
                label = f"WRONG WAY + SIGNAL JUMP {track_id}"
            elif has_signal_jump:
                # Signal jump only = Orange box
                color = (0, 165, 255)
                label = f"SIGNAL JUMP {track_id}"
            elif has_wrong_way:
                # Wrong way only = Red box
                color = (0, 0, 255)
                label = f"WRONG WAY {track_id}"

            cv2.rectangle(frame, (l, t), (l + w, t + h), color, 2)
            cv2.putText(frame, label, (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw counting line
        cv2.line(frame, (0, COUNTING_LINE), (width, COUNTING_LINE), (255, 255, 0), 2)

        # ===== SIGNAL VISUALIZATION =====
        current_signal = get_signal_state(signal_start_time)
        
        # Draw signal zone
        signal_zone_color = (0, 255, 0) if current_signal == "GREEN" else (0, 0, 255)
        cv2.rectangle(frame, (0, SIGNAL_ZONE_Y_START), (width, SIGNAL_ZONE_Y_END), signal_zone_color, 3)
        
        # Display signal state
        signal_text = f"SIGNAL: {current_signal}"
        signal_text_color = (0, 255, 0) if current_signal == "GREEN" else (0, 0, 255)
        cv2.putText(frame, signal_text, (width - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, signal_text_color, 3)
        
        # Display signal jumping count
        cv2.putText(frame, f"Signal Jumped: {signal_jumping_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        
        # Display wrong way count
        cv2.putText(frame, f"Wrong Way: {wrong_way_count}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # ===== ALERT MESSAGES =====
        # Show WRONG WAY alert
        if frame_has_wrong_way:
            cv2.rectangle(frame, (200, 20), (width - 200, 120), (0, 0, 255), 8)
            cv2.putText(frame, "WRONG WAY DETECTED!", (250, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 4)
        
        # Show SIGNAL JUMP alert
        if frame_has_signal_jump:
            cv2.rectangle(frame, (200, 20), (width - 200, 120), (0, 165, 255), 8)
            cv2.putText(frame, "SIGNAL JUMP DETECTED!", (220, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 165, 255), 4)

        out.write(frame)
        cv2.imshow("Wrong Way Detection", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    out.release()
    
    print(f"\n✅ Completed: {video_file}")
    print(f"   Output: {output_file}")
    print(f"   Total vehicles: {vehicle_count}")
    print(f"   Wrong way violations: {wrong_way_count}")
    print(f"   Signal jump violations: {signal_jumping_count}\n")

# Process all videos
for video in VIDEO_FILES:
    process_video(video)

cv2.destroyAllWindows()
print("✅ All videos processed!")

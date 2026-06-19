import cv2
import mediapipe as mp
import numpy as np
import time

# Persistent trail buffer for Preset 4 ghost trails
trail_buffer = None
import math

def is_convex(pts):
    n = len(pts)
    sign = 0
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        dx1 = p2[0] - p1[0]
        dy1 = p2[1] - p1[1]
        dx2 = p3[0] - p2[0]
        dy2 = p3[1] - p2[1]
        cross = dx1 * dy2 - dy1 * dx2
        if abs(cross) > 10.0:
            current_sign = 1 if cross > 0 else -1
            if sign == 0:
                sign = current_sign
            elif sign != current_sign:
                return False
    return True

# -------------------------
# MediaPipe Setup
# -------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# -------------------------
# Mock Classes for persistence & smoothing
# -------------------------
class MockLandmark:
    def __init__(self, x, y, z, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class MockLandmarkList:
    def __init__(self, landmarks):
        self.landmark = landmarks

class MockResults:
    def __init__(self, pose_landmarks, left_hand_landmarks, right_hand_landmarks):
        self.pose_landmarks = pose_landmarks
        self.left_hand_landmarks = left_hand_landmarks
        self.right_hand_landmarks = right_hand_landmarks

def draw_hand_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    # Joint connections
    hand_conns = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),(0,5),(0,17)
    ]
    # Draw green lines (#00e676 -> BGR: (118, 230, 0))
    for a, b in hand_conns:
        pt1 = (int(landmarks.landmark[a].x * w), int(landmarks.landmark[a].y * h))
        pt2 = (int(landmarks.landmark[b].x * w), int(landmarks.landmark[b].y * h))
        cv2.line(frame, pt1, pt2, (118, 230, 0), 2)
    # Draw white points
    for lm in landmarks.landmark:
        pt = (int(lm.x * w), int(lm.y * h))
        cv2.circle(frame, pt, 3, (255, 255, 255), -1)

def draw_pose_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    pose_conns = [
        (11,12),(11,13),(13,15),(12,14),(14,16), # arms
        (11,23),(12,24),(23,24), # torso
        (23,25),(25,27),(24,26),(26,28) # legs
    ]
    # Draw green lines (#00e676 -> BGR: (118, 230, 0))
    for a, b in pose_conns:
        if a < len(landmarks.landmark) and b < len(landmarks.landmark):
            pt1 = (int(landmarks.landmark[a].x * w), int(landmarks.landmark[a].y * h))
            pt2 = (int(landmarks.landmark[b].x * w), int(landmarks.landmark[b].y * h))
            cv2.line(frame, pt1, pt2, (118, 230, 0), 2)
    # Draw red/orange points (#ff3d00 -> BGR: (0, 61, 255))
    for lm in landmarks.landmark:
        if getattr(lm, 'visibility', 1.0) > 0.5:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, pt, 3, (0, 61, 255), -1)

# -------------------------
# Webcam Setup
# -------------------------
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# -------------------------
# Toggle States
# -------------------------
show_pose = False
show_hands = False
flip_h = False
flip_v = False
active_preset = 1
show_outline = False
last_left_hand = None
last_right_hand = None
last_pose = None

# Preset 6 video/image variables
overlay_url = "sample.mp4"
overlay_cap = None
overlay_img = None
overlay_flip_h = False
overlay_flip_v = False

# -------------------------
# Main Loop
# -------------------------
prev_time = 0

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while True:
        loop_start = time.time()

        success, frame = cap.read()
        
        # Calculate FPS
        current_time = time.time()
        fps = int(1 / (current_time - prev_time)) if prev_time > 0 else 0
        prev_time = current_time

        if not success:
            print("Failed to read webcam.")
            break
        
        h, w, _ = frame.shape

        # Mirror / Flip image
        if flip_h and flip_v:
            frame = cv2.flip(frame, -1)
        elif flip_h:
            frame = cv2.flip(frame, 1)
        elif flip_v:
            frame = cv2.flip(frame, 0)

        # Convert BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        raw_results = holistic.process(rgb)

        alpha = 0.25  # Smoothing factor

        # Smooth & Persist Left Hand
        current_lh = raw_results.left_hand_landmarks
        if current_lh:
            if last_left_hand is None:
                last_left_hand = MockLandmarkList([MockLandmark(lm.x, lm.y, lm.z, getattr(lm, 'visibility', 1.0)) for lm in current_lh.landmark])
            else:
                for i in range(21):
                    last_left_hand.landmark[i].x += (current_lh.landmark[i].x - last_left_hand.landmark[i].x) * alpha
                    last_left_hand.landmark[i].y += (current_lh.landmark[i].y - last_left_hand.landmark[i].y) * alpha
                    last_left_hand.landmark[i].z += (current_lh.landmark[i].z - last_left_hand.landmark[i].z) * alpha
                    last_left_hand.landmark[i].visibility = getattr(current_lh.landmark[i], 'visibility', 1.0)

        # Smooth & Persist Right Hand
        current_rh = raw_results.right_hand_landmarks
        if current_rh:
            if last_right_hand is None:
                last_right_hand = MockLandmarkList([MockLandmark(lm.x, lm.y, lm.z, getattr(lm, 'visibility', 1.0)) for lm in current_rh.landmark])
            else:
                for i in range(21):
                    last_right_hand.landmark[i].x += (current_rh.landmark[i].x - last_right_hand.landmark[i].x) * alpha
                    last_right_hand.landmark[i].y += (current_rh.landmark[i].y - last_right_hand.landmark[i].y) * alpha
                    last_right_hand.landmark[i].z += (current_rh.landmark[i].z - last_right_hand.landmark[i].z) * alpha
                    last_right_hand.landmark[i].visibility = getattr(current_rh.landmark[i], 'visibility', 1.0)

        # Smooth & Persist Pose
        current_pose = raw_results.pose_landmarks
        if current_pose:
            if last_pose is None:
                last_pose = MockLandmarkList([MockLandmark(lm.x, lm.y, lm.z, getattr(lm, 'visibility', 1.0)) for lm in current_pose.landmark])
            else:
                for i in range(len(last_pose.landmark)):
                    last_pose.landmark[i].x += (current_pose.landmark[i].x - last_pose.landmark[i].x) * alpha
                    last_pose.landmark[i].y += (current_pose.landmark[i].y - last_pose.landmark[i].y) * alpha
                    last_pose.landmark[i].z += (current_pose.landmark[i].z - last_pose.landmark[i].z) * alpha
                    last_pose.landmark[i].visibility = getattr(current_pose.landmark[i], 'visibility', 1.0)

        results = MockResults(last_pose, last_left_hand, last_right_hand)

        # Preset 2: Reduce exposure of background video by 75%
        if active_preset == 2:
            frame = (frame * 0.25).astype(np.uint8)

        # Preset 4: Fade the trail buffer
        if active_preset == 4:
            if trail_buffer is None or trail_buffer.shape != frame.shape:
                trail_buffer = np.zeros_like(frame)
            trail_buffer = (trail_buffer * 0.85).astype(np.uint8)

        # Preset 5: Full black background
        if active_preset == 5:
            frame = np.zeros_like(frame)


        # -------------------------
        # Draw Pose
        # -------------------------
        if show_pose and results.pose_landmarks:
            draw_pose_landmarks(frame, results.pose_landmarks)

        # -------------------------
        # Draw Left Hand
        # -------------------------
        if show_hands and results.left_hand_landmarks:
            draw_hand_landmarks(frame, results.left_hand_landmarks)

        # -------------------------
        # Draw Right Hand
        # -------------------------
        if show_hands and results.right_hand_landmarks:
            draw_hand_landmarks(frame, results.right_hand_landmarks)

        # -------------------------
        # Corner-pinned White Rectangle (Preset 1 Only)
        # -------------------------
        if active_preset == 1 and results.left_hand_landmarks and results.right_hand_landmarks:
            h, w, _ = frame.shape
            
            # Get landmarks (4 = THUMB_TIP, 8 = INDEX_FINGER_TIP)
            lh_thumb = results.left_hand_landmarks.landmark[4]
            lh_index = results.left_hand_landmarks.landmark[8]
            rh_thumb = results.right_hand_landmarks.landmark[4]
            rh_index = results.right_hand_landmarks.landmark[8]
            
            # Convert to pixel coordinates
            pt1 = (int(lh_thumb.x * w), int(lh_thumb.y * h))
            pt2 = (int(lh_index.x * w), int(lh_index.y * h))
            pt3 = (int(rh_index.x * w), int(rh_index.y * h))
            pt4 = (int(rh_thumb.x * w), int(rh_thumb.y * h))
            
            pts = np.array([pt1, pt2, pt3, pt4], np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Draw semi-transparent filled shape
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (255, 255, 255))
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Draw solid white border outline
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 255), thickness=2)

        # Preset 2: Glow Mesh (Connect H1 fingertips to H2 fingertips, color reactive & glow)
        # -------------------------
        if active_preset == 2 and results.left_hand_landmarks and results.right_hand_landmarks:
            h, w, _ = frame.shape
            lh = results.left_hand_landmarks.landmark
            rh = results.right_hand_landmarks.landmark
            overlay = frame.copy()
            
            tips = [4, 8, 12, 16, 20]
            for i in tips:
                pt1 = (int(lh[i].x * w), int(lh[i].y * h))
                for j in tips:
                    pt2 = (int(rh[j].x * w), int(rh[j].y * h))
                    
                    # Calculate distance
                    dx = pt1[0] - pt2[0]
                    dy = pt1[1] - pt2[1]
                    d = math.sqrt(dx*dx + dy*dy)
                    
                    # Length reactive thickness and color (red when short, thinner/whiter when far)
                    ratio = min(d / (1468.0 * 0.65), 1.0)
                    glow_thick = max(1, int(8 - 6 * ratio))
                    core_thick = max(1, int(2 - 1 * ratio))
                    
                    # Outer glow color (red to white in BGR)
                    b_glow = int(255 * ratio)
                    g_glow = int(255 * ratio)
                    r_glow = 255
                    
                    # Inner core color (light red to white in BGR)
                    b_core = int(128 + 127 * ratio)
                    g_core = int(128 + 127 * ratio)
                    r_core = 255
                    
                    # 1. Thicker outer glow line
                    cv2.line(overlay, pt1, pt2, (b_glow, g_glow, r_glow), glow_thick)
                    # 2. Thinner inner core line
                    cv2.line(overlay, pt1, pt2, (b_core, g_core, r_core), core_thick)
            
            # Blend overlay to achieve semi-transparent glow lines
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # -------------------------
        # Preset 3: Finger Portal Filters
        # -------------------------
        if active_preset == 3 and results.left_hand_landmarks and results.right_hand_landmarks:
            h, w, _ = frame.shape
            lh = results.left_hand_landmarks.landmark
            rh = results.right_hand_landmarks.landmark

            finger_pairs = [
                (4, 8, 4, 8, 'invert'),
                (8, 12, 8, 12, 'grayscale'),
                (12, 16, 12, 16, 'duotone'),
                (16, 20, 16, 20, 'pixelate'),
            ]

            clean = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            for li1, li2, ri1, ri2, filter_name in finger_pairs:
                pts = np.array([
                    [int(lh[li1].x * w), int(lh[li1].y * h)],
                    [int(lh[li2].x * w), int(lh[li2].y * h)],
                    [int(rh[ri2].x * w), int(rh[ri2].y * h)],
                    [int(rh[ri1].x * w), int(rh[ri1].y * h)],
                ], np.int32)

                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, pts, 255)

                filtered = clean.copy()
                if filter_name == 'invert':
                    filtered = 255 - filtered
                elif filter_name == 'grayscale':
                    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
                    filtered = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                elif filter_name == 'duotone':
                    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                    b_ch = np.full_like(gray, 255.0)
                    g_ch = 255.0 * (1.0 - gray)
                    r_ch = 255.0 * gray
                    filtered = np.stack([b_ch, g_ch, r_ch], axis=-1).astype(np.uint8)
                elif filter_name == 'pixelate':
                    block = 8
                    small = cv2.resize(filtered, (w // block, h // block), interpolation=cv2.INTER_LINEAR)
                    filtered = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

                mask_3ch = cv2.merge([mask, mask, mask])
                frame = np.where(mask_3ch == 255, filtered, frame)
                if show_outline:
                    cv2.polylines(frame, [pts], True, (255, 255, 255), 2)

        # -------------------------
        # Preset 4: Ghost Trails
        # -------------------------
        if active_preset == 4 and (results.left_hand_landmarks or results.right_hand_landmarks):
            h, w, _ = frame.shape

            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    px, py = int(lm.x * w), int(lm.y * h)
                    cv2.circle(trail_buffer, (px, py), 5, (255, 255, 0), -1)
                    cv2.circle(frame, (px, py), 6, (255, 255, 0), -1)

            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    px, py = int(lm.x * w), int(lm.y * h)
                    cv2.circle(trail_buffer, (px, py), 5, (128, 0, 255), -1)
                    cv2.circle(frame, (px, py), 6, (128, 0, 255), -1)

            cv2.addWeighted(trail_buffer, 0.7, frame, 1.0, 0, frame)

        # -------------------------
        # Preset 5: Void Skeleton
        # -------------------------
        if active_preset == 5:
            h, w, _ = frame.shape
            elapsed = time.time()
            pulse_r = int(4 + 3 * math.sin(elapsed * 4))

            hand_conns = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),
                (5,9),(9,13),(13,17)
            ]
            fingertips = [4, 8, 12, 16, 20]

            if results.left_hand_landmarks:
                lh = results.left_hand_landmarks.landmark
                overlay = frame.copy()
                for a, b in hand_conns:
                    p1 = (int(lh[a].x * w), int(lh[a].y * h))
                    p2 = (int(lh[b].x * w), int(lh[b].y * h))
                    cv2.line(overlay, p1, p2, (255, 255, 0), 8)
                cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                for a, b in hand_conns:
                    p1 = (int(lh[a].x * w), int(lh[a].y * h))
                    p2 = (int(lh[b].x * w), int(lh[b].y * h))
                    cv2.line(frame, p1, p2, (255, 255, 0), 3)
                for i in range(21):
                    r = pulse_r if i in fingertips else 3
                    pt = (int(lh[i].x * w), int(lh[i].y * h))
                    cv2.circle(frame, pt, r, (255, 255, 0), -1)

            if results.right_hand_landmarks:
                rh = results.right_hand_landmarks.landmark
                overlay = frame.copy()
                for a, b in hand_conns:
                    p1 = (int(rh[a].x * w), int(rh[a].y * h))
                    p2 = (int(rh[b].x * w), int(rh[b].y * h))
                    cv2.line(overlay, p1, p2, (255, 0, 200), 8)
                cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                for a, b in hand_conns:
                    p1 = (int(rh[a].x * w), int(rh[a].y * h))
                    p2 = (int(rh[b].x * w), int(rh[b].y * h))
                    cv2.line(frame, p1, p2, (255, 0, 200), 3)
                for i in range(21):
                    r = pulse_r if i in fingertips else 3
                    pt = (int(rh[i].x * w), int(rh[i].y * h))
                    cv2.circle(frame, pt, r, (255, 0, 200), -1)


        # -------------------------
        # Preset 6: Video/Image Corner-pinned Overlay
        # -------------------------
        if active_preset == 6 and results.left_hand_landmarks and results.right_hand_landmarks:
            frame_o = None
            if overlay_img is not None:
                frame_o = overlay_img.copy()
            elif overlay_cap is not None and overlay_cap.isOpened():
                ret_o, frame_o = overlay_cap.read()
                if not ret_o:
                    overlay_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_o, frame_o = overlay_cap.read()
                
            if frame_o is not None:
                # Apply flips
                if overlay_flip_h and overlay_flip_v:
                    frame_o = cv2.flip(frame_o, -1)
                elif overlay_flip_h:
                    frame_o = cv2.flip(frame_o, 1)
                elif overlay_flip_v:
                    frame_o = cv2.flip(frame_o, 0)

                h_o, w_o, _ = frame_o.shape
                
                # Compare horizontal position of index landmark (0) to identify visual left and right hands
                lh_lms = results.left_hand_landmarks.landmark
                rh_lms = results.right_hand_landmarks.landmark
                
                if lh_lms[0].x < rh_lms[0].x:
                    vL = lh_lms
                    vR = rh_lms
                else:
                    vL = rh_lms
                    vR = lh_lms

                # Target corners between Left & Right Hand's index and thumb tips:
                # pt1 (Left Hand thumb tip)
                # pt2 (Left Hand index tip)
                # pt3 (Right Hand index tip)
                # pt4 (Right Hand thumb tip)
                pt1 = (int(vL[4].x * w), int(vL[4].y * h))
                pt2 = (int(vL[8].x * w), int(vL[8].y * h))
                pt3 = (int(vR[8].x * w), int(vR[8].y * h))
                pt4 = (int(vR[4].x * w), int(vR[4].y * h))
                
                # Source corners of the overlay video/image
                src_pts = np.array([
                    [0, 0],
                    [w_o - 1, 0],
                    [w_o - 1, h_o - 1],
                    [0, h_o - 1]
                ], dtype=np.float32)
                
                # Destination corners: Top-Left (pt2), Top-Right (pt3), Bottom-Right (pt4), Bottom-Left (pt1)
                dst_pts = np.array([
                    pt2,
                    pt3,
                    pt4,
                    pt1
                ], dtype=np.float32)
                
                # Only warp if the destination quad is convex and non-self-intersecting
                if is_convex(dst_pts):
                    # Compute perspective warp homography matrix
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    
                    # Warp overlay frame to fit screen dimensions (w, h)
                    warped_overlay = cv2.warpPerspective(frame_o, M, (w, h))
                    
                    # Create binary mask of the destination quad area
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, np.array([pt2, pt3, pt4, pt1], dtype=np.int32), 255)
                    
                    # Blend the warped media into the main frame
                    mask_3ch = cv2.merge([mask, mask, mask])
                    frame = np.where(mask_3ch == 255, warped_overlay, frame)

        # -------------------------
        # Status Text
        # -------------------------
        cv2.putText(
            frame,
            f"FPS: {fps}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # Draw top-right presets HUD
        start_x = w - 380
        for pi in range(1, 7):
            box_x = start_x + (pi - 1) * 50
            color_box = (0, 255, 0) if active_preset == pi else (100, 100, 100)
            thick_box = -1 if active_preset == pi else 2
            cv2.rectangle(frame, (box_x, 20), (box_x + 40, 60), color_box, thick_box)
            text_col = (0, 0, 0) if active_preset == pi else (255, 255, 255)
            cv2.putText(frame, str(pi), (box_x + 13, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_col, 2)

        # Draw Outline Toggle Box (O) only when Preset 3 is active
        if active_preset == 3:
            box_x = start_x + 6 * 50
            color_box = (0, 255, 0) if show_outline else (100, 100, 100)
            thick_box = -1 if show_outline else 2
            cv2.rectangle(frame, (box_x, 20), (box_x + 40, 60), color_box, thick_box)
            text_col = (0, 0, 0) if show_outline else (255, 255, 255)
            cv2.putText(frame, "O", (box_x + 13, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_col, 2)

        # Show frame
        cv2.imshow("MediaPipe Holistic Tracking", frame)

        # -------------------------
        # Keyboard Controls
        # -------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        elif key == ord('h'):
            show_hands = not show_hands
            print(f"Hands: {'ON' if show_hands else 'OFF'}")

        elif key == ord('o') and active_preset == 3:
            show_outline = not show_outline
            print(f"Outline: {'ON' if show_outline else 'OFF'}")

        elif key == ord('p'):
            show_pose = not show_pose
            print(f"Pose: {'ON' if show_pose else 'OFF'}")

        elif key == ord('x'):
            flip_h = not flip_h
            print(f"Flip H: {'ON' if flip_h else 'OFF'}")

        elif key == ord('y'):
            flip_v = not flip_v
            print(f"Flip V: {'ON' if flip_v else 'OFF'}")

        elif key == ord('1'):
            active_preset = 1
            print("Preset 1 activated")

        elif key == ord('2'):
            active_preset = 2
            print("Preset 2 activated")

        elif key == ord('3'):
            active_preset = 3
            print("Preset 3 activated")

        elif key == ord('4'):
            active_preset = 4
            trail_buffer = None
            print("Preset 4 activated")

        elif key == ord('5'):
            active_preset = 5
            print("Preset 5 activated")

        elif key == ord('6'):
            active_preset = 6
            print("Preset 6 activated")
            if overlay_cap is None and overlay_img is None:
                import os
                if overlay_url == "sample.mp4" and not os.path.exists("sample.mp4"):
                    print("Default video 'sample.mp4' not found. Downloading...")
                    import subprocess
                    try:
                        subprocess.run(["curl", "-L", "-o", "sample.mp4", "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4"], check=True)
                        print("Download complete.")
                    except Exception as e:
                        print(f"Failed to download default video: {e}")
                
                if overlay_url.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                    overlay_img = cv2.imread(overlay_url)
                else:
                    overlay_cap = cv2.VideoCapture(overlay_url)

        elif key == ord('u') and active_preset == 6:
            print("\n--- Video/Image Source Update ---")
            print("Camera feed paused. A file chooser dialog has opened.")
            file_path = None
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                file_path = filedialog.askopenfilename(
                    title="Select Video or Image for Overlay",
                    filetypes=[
                        ("All Supported Media", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.webp *.bmp"),
                        ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                        ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp")
                    ]
                )
                root.destroy()
            except Exception as e:
                print(f"Failed to open GUI file dialog ({e}). Fallback to terminal input.")
                file_path = input("Enter path to local video or image: ").strip()
            
            if file_path:
                import os
                if os.path.exists(file_path):
                    overlay_url = file_path
                    if overlay_cap is not None:
                        overlay_cap.release()
                        overlay_cap = None
                    overlay_img = None
                    
                    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                        overlay_img = cv2.imread(file_path)
                        if overlay_img is not None:
                            print(f"Loaded local image: {file_path}")
                        else:
                            print(f"Failed to read image: {file_path}")
                    else:
                        overlay_cap = cv2.VideoCapture(file_path)
                        if overlay_cap.isOpened():
                            print(f"Loaded local video: {file_path} (playing on repeat)")
                        else:
                            print(f"Failed to open video: {file_path}")
                else:
                    print(f"File not found: {file_path}")
            print("Resuming camera feed.\n")

        elif key == ord('[') and active_preset == 6:
            overlay_flip_h = not overlay_flip_h
            print(f"Overlay Flip H: {'ON' if overlay_flip_h else 'OFF'}")

        elif key == ord(']') and active_preset == 6:
            overlay_flip_v = not overlay_flip_v
            print(f"Overlay Flip V: {'ON' if overlay_flip_v else 'OFF'}")

        # Lock to 25 FPS (40ms interval)
        elapsed = time.time() - loop_start
        remaining = 0.04 - elapsed
        if remaining > 0:
            time.sleep(remaining)



# -------------------------
# Cleanup
# -------------------------
cap.release()
if overlay_cap is not None:
    overlay_cap.release()
cv2.destroyAllWindows()
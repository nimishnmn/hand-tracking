import cv2
import mediapipe as mp
import numpy as np
import time

# Persistent trail buffer for Preset 4 ghost trails
trail_buffer = None
import math

# -------------------------
# MediaPipe Setup
# -------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

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

# -------------------------
# Main Loop
# -------------------------
prev_time = 0

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while True:

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
        results = holistic.process(rgb)

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
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS
            )

        # -------------------------
        # Draw Left Hand
        # -------------------------
        if show_hands and results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        # -------------------------
        # Draw Right Hand
        # -------------------------
        if show_hands and results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

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
                    
                    # Color reactive (Cyan to Magenta gradient in BGR)
                    ratio = min(d / (1468.0 * 0.65), 1.0)
                    b = 255
                    g = int(255 * (1.0 - ratio))
                    r = int(255 * ratio)
                    
                    # 1. Thicker outer glow line
                    cv2.line(overlay, pt1, pt2, (b, g, r), 4)
                    # 2. Thinner inner core line
                    cv2.line(overlay, pt1, pt2, (255, 255, 255), 1)
            
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
        start_x = w - 320
        for pi in range(1, 6):
            box_x = start_x + (pi - 1) * 50
            color_box = (0, 255, 0) if active_preset == pi else (100, 100, 100)
            thick_box = -1 if active_preset == pi else 2
            cv2.rectangle(frame, (box_x, 20), (box_x + 40, 60), color_box, thick_box)
            text_col = (0, 0, 0) if active_preset == pi else (255, 255, 255)
            cv2.putText(frame, str(pi), (box_x + 13, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_col, 2)

        # Draw Outline Toggle Box (O)
        box_x = start_x + 5 * 50
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

        elif key == ord('o'):
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



# -------------------------
# Cleanup
# -------------------------
cap.release()
cv2.destroyAllWindows()
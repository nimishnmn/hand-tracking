import cv2
import mediapipe as mp
import numpy as np
import time
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

        # -------------------------
        # Preset 2: Glow Mesh (Connect H1 all points to H2 all points, color reactive & glow)
        # -------------------------
        if active_preset == 2 and results.left_hand_landmarks and results.right_hand_landmarks:
            h, w, _ = frame.shape
            lh = results.left_hand_landmarks.landmark
            rh = results.right_hand_landmarks.landmark
            overlay = frame.copy()
            
            for i in range(21):
                pt1 = (int(lh[i].x * w), int(lh[i].y * h))
                for j in range(21):
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
                    
                    cv2.line(overlay, pt1, pt2, (b, g, r), 1)
            
            # Blend overlay to achieve semi-transparent glow lines
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # -------------------------
        # Status Text
        # -------------------------
        cv2.putText(
            frame,
            "ESC = Quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # Draw top-right presets HUD
        # Box 1
        color1 = (0, 255, 0) if active_preset == 1 else (100, 100, 100)
        thick1 = -1 if active_preset == 1 else 2
        cv2.rectangle(frame, (1180, 20), (1220, 60), color1, thick1)
        textColor1 = (0, 0, 0) if active_preset == 1 else (255, 255, 255)
        cv2.putText(frame, "1", (1193, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, textColor1, 2)
        
        # Box 2
        color2 = (0, 255, 0) if active_preset == 2 else (100, 100, 100)
        thick2 = -1 if active_preset == 2 else 2
        cv2.rectangle(frame, (1230, 20), (1270, 60), color2, thick2)
        textColor2 = (0, 0, 0) if active_preset == 2 else (255, 255, 255)
        cv2.putText(frame, "2", (1243, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, textColor2, 2)

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



# -------------------------
# Cleanup
# -------------------------
cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import time

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

        # Mirror image
        frame = frame

        # Convert BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = holistic.process(rgb)


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
        # Corner-pinned White Rectangle
        # -------------------------
        if results.left_hand_landmarks and results.right_hand_landmarks:
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
        # Status Text
        # -------------------------
        cv2.putText(
            frame,
            f"Pose [P]: {'ON' if show_pose else 'OFF'}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Hands [H]: {'ON' if show_hands else 'OFF'}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "ESC = Quit",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

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



# -------------------------
# Cleanup
# -------------------------
cap.release()
cv2.destroyAllWindows()
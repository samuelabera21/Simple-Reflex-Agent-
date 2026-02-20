import cv2
import time
from src.detector import FaceDetector, FaceMeshDetector, HandDetector
from src.utils import analyze_hands, count_hands, count_total_fingers

def main():
    cap = cv2.VideoCapture(0)
    hand_detector = HandDetector()
    face_detector = FaceDetector()
    face_mesh_detector = FaceMeshDetector()
    previous_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        hand_results = hand_detector.detect(frame)
        hand_detector.draw(frame, hand_results)

        face_results = face_detector.detect(frame)
        face_count = face_detector.draw(frame, face_results)

        mesh_results = face_mesh_detector.detect(frame)
        face_mesh_detector.draw_mouth(frame, mesh_results)
        mouth_ratio = face_mesh_detector.mouth_ratio(mesh_results, frame.shape)
        teeth_visible, teeth_score = face_mesh_detector.detect_teeth(frame, mesh_results)
        if mouth_ratio is None:
            mouth_status = "No Face"
            mouth_color = (200, 200, 200)
        elif mouth_ratio > 0.35 and teeth_visible:
            mouth_status = "You are laughing"
            mouth_color = (0, 255, 255)
        elif mouth_ratio > 0.35:
            mouth_status = "Mouth open"
            mouth_color = (0, 200, 255)
        elif mouth_ratio > 0.22:
            mouth_status = "You are smiling"
            mouth_color = (0, 255, 0)
        else:
            mouth_status = "You closed your mouth"
            mouth_color = (0, 0, 255)

        if teeth_visible is None:
            teeth_status = "Teeth: N/A"
            teeth_color = (200, 200, 200)
        elif teeth_visible:
            teeth_status = f"Teeth: Visible ({int(teeth_score * 100)}%)"
            teeth_color = (255, 255, 255)
        else:
            teeth_status = "Teeth: Not visible"
            teeth_color = (180, 180, 180)

        num_hands = count_hands(hand_results)
        analyses = analyze_hands(hand_results)
        total_fingers = count_total_fingers(analyses)

        current_time = time.time()
        fps = 1 / (current_time - previous_time) if current_time != previous_time else 0
        previous_time = current_time

        cv2.putText(frame, f'Hands: {num_hands}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f'Total Fingers: {total_fingers}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.putText(frame, f'Faces: {face_count}', (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        cv2.putText(frame, f'FPS: {int(fps)}', (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, mouth_status, (10, 210),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, mouth_color, 2)

        cv2.putText(frame, teeth_status, (10, 245),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, teeth_color, 2)

        cv2.putText(frame, 'Press S or ESC to quit', (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        y_pos = 280
        for hand_info in analyses:
            text = (
                f"H{hand_info['hand_index']} {hand_info['label']}: "
                f"{hand_info['finger_count']} fingers, {hand_info['gesture']}"
            )
            cv2.putText(frame, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_pos += 30

        cv2.imshow("Gesture Project", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('s') or key == ord('S'):  # ESC or S to exit
            break

    cap.release()
    hand_detector.close()
    face_detector.close()
    face_mesh_detector.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import cv2
import time

from logger import ActionLogger
from reflex_agent import SimpleReflexAgent
from ui import ProfessionalOverlay
from src.detector import FaceDetector, FaceMeshDetector, HandDetector
from src.utils import analyze_hands, count_hands, count_total_fingers


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    hand_detector = HandDetector()
    face_detector = FaceDetector()
    face_mesh_detector = FaceMeshDetector()

    agent = SimpleReflexAgent()
    overlay = ProfessionalOverlay()
    logger = ActionLogger("actions.log")

    previous_time = time.time()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            hand_results = hand_detector.detect(frame)
            face_results = face_detector.detect(frame)
            mesh_results = face_mesh_detector.detect(frame)

            hand_detector.draw(frame, hand_results)
            face_count = face_detector.draw(frame, face_results)
            face_mesh_detector.draw(frame, mesh_results)

            num_hands = count_hands(hand_results)
            analyses = analyze_hands(hand_results)
            total_fingers = count_total_fingers(analyses)

            mouth_status = face_mesh_detector.classify_mouth_state(mesh_results, frame.shape)
            teeth_visible, teeth_score = face_mesh_detector.detect_teeth(frame, mesh_results)

            if teeth_visible is None:
                teeth_status = "N/A"
            elif teeth_visible:
                teeth_status = f"Visible ({int(teeth_score * 100)}%)"
            else:
                teeth_status = "Not visible"

            action = agent.decide(total_fingers, face_count, mouth_status)
            logger.log_action(action)

            current_time = time.time()
            fps = 1.0 / (current_time - previous_time) if current_time != previous_time else 0.0
            previous_time = current_time

            overlay.draw(
                frame,
                {
                    "fps": fps,
                    "hands": num_hands,
                    "fingers": total_fingers,
                    "faces": face_count,
                    "mouth_status": mouth_status,
                    "teeth_status": teeth_status,
                    "action": action,
                },
            )

            cv2.imshow("GestureSense", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("s"), ord("S")):
                break
            if key in (ord("r"), ord("R")):
                logger.log_action("Manual Reset")
            if key in (ord("l"), ord("L")):
                logger.clear_logs()
    finally:
        cap.release()
        hand_detector.close()
        face_detector.close()
        face_mesh_detector.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
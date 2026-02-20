import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return results

    def draw(self, frame, results):
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, handLms, self.mp_hands.HAND_CONNECTIONS
                )

    def close(self):
        self.hands.close()


class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_detector.process(rgb)

    def draw(self, frame, results):
        detections = results.detections or []
        frame_height, frame_width = frame.shape[:2]

        for detection in detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * frame_width)
            y = int(bbox.ymin * frame_height)
            w = int(bbox.width * frame_width)
            h = int(bbox.height * frame_height)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            score = int(detection.score[0] * 100)
            cv2.putText(
                frame,
                f"Face {score}%",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )

        return len(detections)

    def close(self):
        self.face_detector.close()


class FaceMeshDetector:
    def __init__(self, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(rgb)

    def mouth_ratio(self, results, frame_shape):
        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0].landmark
        frame_height, frame_width = frame_shape[:2]

        left_corner = face_landmarks[61]
        right_corner = face_landmarks[291]
        upper_lip = face_landmarks[13]
        lower_lip = face_landmarks[14]

        left_x = left_corner.x * frame_width
        left_y = left_corner.y * frame_height
        right_x = right_corner.x * frame_width
        right_y = right_corner.y * frame_height

        upper_x = upper_lip.x * frame_width
        upper_y = upper_lip.y * frame_height
        lower_x = lower_lip.x * frame_width
        lower_y = lower_lip.y * frame_height

        horizontal = ((right_x - left_x) ** 2 + (right_y - left_y) ** 2) ** 0.5
        vertical = ((lower_x - upper_x) ** 2 + (lower_y - upper_y) ** 2) ** 0.5

        if horizontal == 0:
            return None

        return vertical / horizontal

    def _mouth_points(self, results, frame_shape, indices):
        face_landmarks = results.multi_face_landmarks[0].landmark
        frame_height, frame_width = frame_shape[:2]
        points = []
        for idx in indices:
            pt = face_landmarks[idx]
            points.append((int(pt.x * frame_width), int(pt.y * frame_height)))
        return points

    def draw(self, frame, results, color=(0, 140, 255), thickness=2):
        if not results.multi_face_landmarks:
            return

        outer_lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61]
        inner_lips = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 78]

        outer_points = self._mouth_points(results, frame.shape, outer_lips)
        inner_points = self._mouth_points(results, frame.shape, inner_lips)

        for i in range(len(outer_points) - 1):
            cv2.line(frame, outer_points[i], outer_points[i + 1], color, thickness)

        for i in range(len(inner_points) - 1):
            cv2.line(frame, inner_points[i], inner_points[i + 1], (255, 255, 255), 1)

    def classify_mouth_state(self, results, frame_shape):
        ratio = self.mouth_ratio(results, frame_shape)
        if ratio is None:
            return "No Face"
        if ratio > 0.35:
            return "You are laughing"
        if ratio > 0.22:
            return "You are smiling"
        if ratio > 0.16:
            return "Mouth open"
        return "Mouth closed"

    def detect_teeth(self, frame, results):
        if not results.multi_face_landmarks:
            return None, 0.0

        inner_lips = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 78]
        points = self._mouth_points(results, frame.shape, inner_lips)

        if len(points) < 3:
            return None, 0.0

        points_array = np.array(points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(points_array)
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)

        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return None, 0.0

        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = np.array([(px - x, py - y) for px, py in points], dtype=np.int32)
        cv2.fillPoly(mask, [shifted], 255)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)

        bright = (v_channel > 180).astype(np.uint8) * 255
        low_sat = (s_channel < 60).astype(np.uint8) * 255
        teeth_mask = cv2.bitwise_and(bright, low_sat)
        teeth_mask = cv2.bitwise_and(teeth_mask, mask)

        mouth_pixels = cv2.countNonZero(mask)
        teeth_pixels = cv2.countNonZero(teeth_mask)

        if mouth_pixels == 0:
            return None, 0.0

        score = teeth_pixels / mouth_pixels
        visible = score > 0.08

        return visible, score

    def close(self):
        self.face_mesh.close()
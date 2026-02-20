import cv2


COLOR_GREEN = (80, 220, 130)
COLOR_YELLOW = (0, 210, 255)
COLOR_RED = (70, 70, 255)
COLOR_TEXT = (230, 230, 230)
COLOR_MUTED = (170, 170, 170)
COLOR_ACCENT = (255, 170, 50)


class ProfessionalOverlay:
    def __init__(self, title="GestureSense – Simple Reflex Vision Agent"):
        self.title = title

    @staticmethod
    def _panel(frame, x1, y1, x2, y2, alpha=0.45, color=(20, 20, 20)):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    @staticmethod
    def _text(frame, text, x, y, color=COLOR_TEXT, scale=0.6, thickness=1):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def action_color(action):
        if action in {"Greeting User", "User Happy", "High Engagement"}:
            return COLOR_GREEN
        if action in {"No User Detected", "Stop Interaction"}:
            return COLOR_RED
        return COLOR_YELLOW

    @staticmethod
    def status_color(mouth_status, face_count):
        if face_count == 0:
            return COLOR_RED
        if mouth_status in {"You are smiling", "You are laughing"}:
            return COLOR_GREEN
        if mouth_status == "Mouth open":
            return COLOR_YELLOW
        return COLOR_MUTED

    def draw(self, frame, data):
        h, w = frame.shape[:2]

        self._panel(frame, 12, 12, w - 12, 56, alpha=0.5)
        self._text(frame, self.title, 26, 42, COLOR_ACCENT, scale=0.7, thickness=2)

        left_x1, left_y1, left_x2, left_y2 = 12, 70, 410, min(h - 70, 290)
        right_x1, right_y1, right_x2, right_y2 = max(430, w - 430), 70, w - 12, min(h - 70, 250)
        bottom_x1, bottom_y1, bottom_x2, bottom_y2 = 12, max(300, h - 140), w - 12, h - 12

        self._panel(frame, left_x1, left_y1, left_x2, left_y2)
        self._panel(frame, right_x1, right_y1, right_x2, right_y2)
        self._panel(frame, bottom_x1, bottom_y1, bottom_x2, bottom_y2)

        self._text(frame, "System Info", left_x1 + 14, left_y1 + 28, COLOR_ACCENT, scale=0.62, thickness=2)
        self._text(frame, f"FPS: {data['fps']:.1f}", left_x1 + 14, left_y1 + 58)
        self._text(frame, f"Hands Detected: {data['hands']}", left_x1 + 14, left_y1 + 84)
        self._text(frame, f"Total Fingers: {data['fingers']}", left_x1 + 14, left_y1 + 110)
        self._text(frame, f"Faces Detected: {data['faces']}", left_x1 + 14, left_y1 + 136)
        self._text(frame, f"Mouth State: {data['mouth_status']}", left_x1 + 14, left_y1 + 162, self.status_color(data['mouth_status'], data['faces']))
        self._text(frame, f"Teeth: {data['teeth_status']}", left_x1 + 14, left_y1 + 188, COLOR_MUTED)

        self._text(frame, "Agent Action", right_x1 + 14, right_y1 + 28, COLOR_ACCENT, scale=0.62, thickness=2)
        self._text(frame, data['action'], right_x1 + 14, right_y1 + 70, self.action_color(data['action']), scale=0.9, thickness=2)

        self._text(frame, "User Status", bottom_x1 + 14, bottom_y1 + 30, COLOR_ACCENT, scale=0.62, thickness=2)
        user_status = "User Present" if data['faces'] > 0 else "Awaiting User"
        self._text(frame, user_status, bottom_x1 + 14, bottom_y1 + 62, self.status_color(data['mouth_status'], data['faces']), scale=0.75, thickness=2)
        self._text(frame, "Controls: S/ESC Exit  |  R Reset  |  L Clear Logs", bottom_x1 + 14, bottom_y2 - 18, COLOR_MUTED, scale=0.55, thickness=1)

class SimpleReflexAgent:
    def decide(self, finger_count, face_count, mouth_status):
        if face_count == 0:
            return "No User Detected"
        if finger_count == 0:
            return "Stop Interaction"
        if finger_count == 5:
            return "Greeting User"
        if mouth_status == "You are smiling":
            return "User Happy"
        if mouth_status == "You are laughing":
            return "High Engagement"
        return "Idle"

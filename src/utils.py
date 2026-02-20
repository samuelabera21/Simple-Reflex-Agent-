TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [2, 6, 10, 14, 18]


def count_hands(results):
    if results.multi_hand_landmarks:
        return len(results.multi_hand_landmarks)
    return 0


def _is_thumb_up(landmarks, handedness_label):
    thumb_tip_x = landmarks[TIP_IDS[0]].x
    thumb_joint_x = landmarks[PIP_IDS[0]].x

    if handedness_label == "Right":
        return thumb_tip_x < thumb_joint_x
    return thumb_tip_x > thumb_joint_x


def _is_finger_up(landmarks, tip_id, pip_id):
    return landmarks[tip_id].y < landmarks[pip_id].y


def analyze_hands(results):
    analyses = []

    if not results.multi_hand_landmarks:
        return analyses

    handedness_list = results.multi_handedness or []

    for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
        landmarks = hand_landmarks.landmark

        label = "Unknown"
        score = 0.0
        if index < len(handedness_list):
            classification = handedness_list[index].classification[0]
            label = classification.label
            score = classification.score

        fingers_up = [
            _is_thumb_up(landmarks, label),
            _is_finger_up(landmarks, TIP_IDS[1], PIP_IDS[1]),
            _is_finger_up(landmarks, TIP_IDS[2], PIP_IDS[2]),
            _is_finger_up(landmarks, TIP_IDS[3], PIP_IDS[3]),
            _is_finger_up(landmarks, TIP_IDS[4], PIP_IDS[4]),
        ]

        finger_count = sum(fingers_up)

        if finger_count == 0:
            gesture = "Fist"
        elif finger_count == 5:
            gesture = "Open Palm"
        else:
            gesture = "Partial"

        analyses.append(
            {
                "hand_index": index + 1,
                "label": label,
                "confidence": score,
                "fingers_up": fingers_up,
                "finger_count": finger_count,
                "gesture": gesture,
            }
        )

    return analyses


def count_total_fingers(analyses):
    return sum(hand_info["finger_count"] for hand_info in analyses)
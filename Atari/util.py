import cv2

def process_frame(frame):
    # Expecting the frame to be (210, 160)

    frame = cv2.resize(frame, (84, 110))
    frame = frame[17:101, :]

    return frame / 255.0
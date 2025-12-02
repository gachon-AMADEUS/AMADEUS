import cv2, os

video = cv2.VideoCapture("찬우발.mp4")
os.makedirs("foot_frames", exist_ok=True)

i = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    cv2.imwrite(f"foot_frames/frame_{i:04d}.jpg", frame)
    i += 1

import cv2
from ultralytics import YOLO  # type: ignore

model = YOLO("yolov8n.pt")
video_path = "./video/crop.mov"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

original_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

frame_width, frame_height = original_frame_height, original_frame_width

output_path = "results/result.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

DOOR_THRESHOLD_X = 390

last_x_positions = {}
processed_ids = set()
alighting_count = 0
boarding_count = 0

COLOR_ALIGHTING = (0, 0, 255)
COLOR_BOARDING = (0, 255, 0)
COLOR_UNDETERMINED = (128, 128, 128)
COLOR_WHITE = (255, 255, 255)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

    for result in results:
        if result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
            if cls != 0:
                continue

            x1, y1, x2, y2 = box
            x_center = (x1 + x2) // 2

            box_color = COLOR_WHITE
            direction_label = ""

            if track_id not in last_x_positions:
                last_x_positions[track_id] = x_center

            elif track_id not in processed_ids:

                prev_x = last_x_positions[track_id]
                curr_x = x_center

                last_x_positions[track_id] = curr_x

                # ALIGHTING (Moving Left)
                if prev_x >= DOOR_THRESHOLD_X and curr_x < DOOR_THRESHOLD_X:
                    alighting_count += 1
                    processed_ids.add(track_id)

                # BOARDING (Moving Right)
                elif prev_x < DOOR_THRESHOLD_X and curr_x >= DOOR_THRESHOLD_X:
                    boarding_count += 1
                    processed_ids.add(track_id)

            if track_id in processed_ids:
                if x_center < DOOR_THRESHOLD_X:
                    box_color = COLOR_ALIGHTING
                    direction_label = "Alighted"
                else:
                    box_color = COLOR_BOARDING
                    direction_label = "Boarded"
            else:
                box_color = COLOR_UNDETERMINED
                direction_label = "Tracking"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            label = f"{direction_label} | {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                2,
            )

    cv2.line(
        frame,
        (DOOR_THRESHOLD_X, 0),
        (DOOR_THRESHOLD_X, frame_height),
        COLOR_WHITE,
        2,
    )
    cv2.putText(
        frame,
        f"ALIGHTING : {alighting_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_WHITE,
        2,
    )
    cv2.putText(
        frame,
        f"BOARDING: {boarding_count}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_WHITE,
        2,
    )

    out.write(frame)
    cv2.imshow("YOLOv8 Direction Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

import cv2


def put_data_on_image(image, cls_name, conf, boxes):
    """
    :param image: numpy array
    :param cls_name: name of class
    :param conf: confidence
    :param boxes: box coords
    :return:
    """
    cv2.rectangle(image, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (0, 0, 255), 5)
    x_text_coord = int(boxes[0])
    y_text_coord = int(boxes[1]) - 10 if int(boxes[1]) >= 10 else 0
    cv2.putText(image, f'{cls_name} {conf:.2f}', (x_text_coord, y_text_coord),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)


def perform_detection(video_path, out_path, model):
    """
    method for read video, perform detection on video and save output video with boxes
    in out_path

    :param video_path: path to input video
    :param out_path: path to output video
    :param model: YOLO model
    :return:
    """
    results = model.predict(video_path, stream=True)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_index = 0
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            results.close()
            break
        frame_results = next(results)
        for res in frame_results:
            data = res.boxes.data.cpu().numpy().squeeze()
            cls = int(data[-1])
            cls_name = res.names[cls]
            conf = data[-2]
            boxes = data[0:4]
            if cls == 0:
                put_data_on_image(image, cls_name, conf, boxes)
        out.write(image)
        frame_index += 1
    cap.release()
    out.release()

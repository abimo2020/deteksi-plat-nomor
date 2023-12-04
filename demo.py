import argparse

import os
import numpy as np
import cv2 as cv
import pytesseract
from datetime import datetime
import mysql.connector

from lpd_yunet import LPD_YuNet

# Check OpenCV version
assert cv.__version__ >= "4.8.0", \
       "Please install latest opencv-python to try this demo: python3 -m pip install --upgrade opencv-python"

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(description='LPD-YuNet for License Plate Detection')
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set path to the input image. Omit for using default camera.')
parser.add_argument('--model', '-m', type=str, default='license_plate_detection_lpd_yunet_2023mar.onnx',
                    help='Usage: Set model path, defaults to license_plate_detection_lpd_yunet_2023mar.onnx.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--conf_threshold', type=float, default=0.9,
                    help='Usage: Set the minimum needed confidence for the model to identify a license plate, defaults to 0.9. Smaller values may result in faster detection, but will limit accuracy. Filter out faces of confidence < conf_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3,
                    help='Usage: Suppress bounding boxes of iou >= nms_threshold. Default = 0.3. Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000,
                    help='Usage: Keep top_k bounding boxes before NMS.')
parser.add_argument('--keep_top_k', type=int, default=750,
                    help='Usage: Keep keep_top_k bounding boxes after NMS.')
parser.add_argument('--save', '-s', action='store_true',
                    help='Usage: Specify to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Usage: Specify to open a new window to show results. Invalid in case of camera input.')
parser.add_argument('--output_directory', type=str, default='result', help='Usage: Set the output directory for captured frames, defaults to captured_frames.')
parser.add_argument('--host', type=str, default='localhost',
                    help='MySQL host, defaults to localhost.')
parser.add_argument('--user', type=str, default='root',
                    help='MySQL username.')
parser.add_argument('--password', type=str, default='popo1212',
                    help='MySQL password.')
parser.add_argument('--database', type=str, default='cv',
                    help='MySQL database name.')
args = parser.parse_args()

def visualize(image, dets, line_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    output = image.copy()
    plate_text = ''

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    for det in dets:
        bbox = det[:-1].astype(np.int32)
        x1, y1, x2, y2, x3, y3, x4, y4 = bbox

        # Draw the border of license plate
        cv.line(output, (x1, y1), (x2, y2), line_color, 2)
        cv.line(output, (x2, y2), (x3, y3), line_color, 2)
        cv.line(output, (x3, y3), (x4, y4), line_color, 2)
        cv.line(output, (x4, y4), (x1, y1), line_color, 2)

        # Crop the license plate region
        plate_roi = image[y1:y3, x1:x3]

        # Perform OCR on the license plate region
        plate_text = pytesseract.image_to_string(plate_roi, config='--psm 8')
        
        cv.putText(output, 'Plat Nomor : {}'.format(plate_text), (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    return output,plate_text

if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    # Instantiate LPD-YuNet
    model = LPD_YuNet(modelPath=args.model,
                      confThreshold=args.conf_threshold,
                      nmsThreshold=args.nms_threshold,
                      topK=args.top_k,
                      keepTopK=args.keep_top_k,
                      backendId=backend_id,
                      targetId=target_id)
    
    # Connect to the MySQL database
    conn = mysql.connector.connect(
        host=args.host,
        user=args.user,
        password=args.password,
        database=args.database
    )
    cursor = conn.cursor()

    # Create a table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plat_nomor (
            id INT AUTO_INCREMENT PRIMARY KEY,
            text VARCHAR(255),
            image_path VARCHAR(255),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()

    # If input is an image
    if args.input is not None:
        image = cv.imread(args.input)
        h, w, _ = image.shape

        # Inference
        model.setInputSize([w, h])
        results = model.infer(image)

        # Print results
        print('{} license plates detected.'.format(results.shape[0]))

        # Draw results on the input image
        image = visualize(image, results)

        # Save results if save is true
        if args.save:
            print('Resutls saved to result.jpg')
            cv.imwrite('result.jpg', image)

        # Visualize results in a new window
        if args.vis:
            cv.namedWindow(args.input, cv.WINDOW_AUTOSIZE)
            cv.imshow(args.input, image)
            cv.waitKey(0)
    else: # Omit input to call default camera
        deviceId = 0
        cap = cv.VideoCapture(deviceId)
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        model.setInputSize([w, h])

        if args.output_directory and not os.path.exists(args.output_directory):
                os.makedirs(args.output_directory)

        tm = cv.TickMeter()
        while True :
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            # Inference
            tm.start()
            results = model.infer(frame) # results is a tuple
            tm.stop()

            # Draw results on the input image
            frame,plate_text = visualize(frame, results, fps=tm.getFPS())
            key = cv.waitKey(1)
            if key == ord('p') or key == ord('P'):
                if args.output_directory and len(results) > 0:
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]  # Format: YYYYMMDDHHMMSSmmm
                    filename = os.path.join(args.output_directory, f'captured_frame_{timestamp}.jpg')
                    cv.imwrite(filename, frame)
                    print(f'Frame captured: {filename}')
                    cursor.execute('INSERT INTO plat_nomor (text, image_path, timestamp) VALUES (%s, %s, %s)',
                       (plate_text, filename, datetime.now()))
                    conn.commit()
            elif key == ord('q') or key == ord('Q'):
                break  

            # Visualize results in a new Window
            cv.imshow('LPD-YuNet Demo', frame)

            tm.reset()

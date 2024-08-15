import cv2
import torch
import numpy as np
import struct
import socket
from deep_sort_realtime.deepsort_tracker import DeepSort
# import paramiko

class YoloDetector():
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                    path='./yolov5s.pt', force_reload=True)
        self.model.classes = [0]  # 'person' 클래스만 탐지
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
    def score_frame(self, frame):
        self.model.to(self.device)
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord
    def plot_boxes(self, results, frame, height, width, confidence=0.3):
        labels, cord = results
        detections = []
        for i in range(len(labels)):
            row = cord[i]
            if row[4] >= confidence:
                x1, y1, x2, y2 = int(row[0]*width), int(row[1]*height), int(row[2]*width), int(row[3]*height)
                if self.model.names[int(labels[i])] == 'person':
                    box_area = (x2 - x1) * (y2 - y1)  # 바운딩박스의 넓이 계산
                    detections.append(([x1, y1, x2, y2, labels[i]], row[4]))
        return frame, detections

# def run_remote_script1():
#     hostname = '192.168.0.8'  # 실제 라즈베리파이의 IP 주소로 변경
#     username = 'ubuntu'                      # 라즈베리파이의 사용자 이름
#     password = '1234567890'           # 라즈베리파이의 비밀번호
#     port = 22                            # SSH 포트 (기본값은 22)
#     client = paramiko.SSHClient()
#     client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#     client.connect(hostname, port=port, username=username, password=password)
#     # 원격 스크립트 실행
#     stdin, stdout, stderr = client.exec_command(f'/home/ubuntu/miniforge3/bin/python3.10 /home/ubuntu/client_tracking_detection.py')
#     # 결과 출력
#     print("STDOUT:")
#     for line in stdout.read().splitlines():
#         print(line.decode('utf-8'))
#     print("STDERR:")
#     for line in stderr.read().splitlines():
#         print(line.decode('utf-8'))
    
#     client.close()

# def run_remote_script2():
#     hostname = '192.168.0.8'  # 실제 라즈베리파이의 IP 주소로 변경
#     username = 'ubuntu'                      # 라즈베리파이의 사용자 이름
#     password = '1234567890'           # 라즈베리파이의 비밀번호
#     port = 22                            # SSH 포트 (기본값은 22)
#     client = paramiko.SSHClient()
#     client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#     client.connect(hostname, port=port, username=username, password=password)
#     # 원격 스크립트 실행
#     stdin, stdout, stderr = client.exec_command(f'/home/ubuntu/miniforge3/bin/python3.10 /home/ubuntu/motor_receive.py')
#     # 결과 출력
#     print("STDOUT:")
#     for line in stdout.read().splitlines():
#         print(line.decode('utf-8'))
#     print("STDERR:")
#     for line in stderr.read().splitlines():
#         print(line.decode('utf-8'))
    
#     client.close()

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def crop_image(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop

def send_direction_and_speed(conn, direction, speed):
    # 방향과 속력을 문자열로 전송
    message = f"{direction},{speed}"
    conn.sendall(message.encode())


def main():
    # 카메라 소켓
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("0.0.0.0", 8181))
    server_socket.listen(1)
    print("Waiting for a connection...")
    # run_remote_script1()
    client_socket, addr = server_socket.accept()
    print(f"Connected to {addr}")

    # 모터 소켓
    command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    command_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    command_socket.bind(("192.168.0.43", 12345))  # 명령을 보낼 클라이언트를 위한 포트
    command_socket.listen()
    print("서버가 연결을 기다리고 있습니다...")
    # run_remote_script2()
    command_conn, command_addr = command_socket.accept()
    print(f"{command_addr}에서 연결되었습니다.")

    detector = YoloDetector()
    tracker = DeepSort(max_age=300)
    first_detected_id = None
    safe_distance = 100
    frame_skip = 3  # 모든 두 번째 프레임을 처리
    frame_count = 0

    while True:
        lengthbuf = recvall(client_socket, 4)
        if lengthbuf is None:
            print("Connection closed by client.")
            break
        length, = struct.unpack('<L', lengthbuf)
        frame_data = recvall(client_socket, length)
        if frame_data is None:
            print("Failed to receive complete frame data.")
            continue
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            print("Could not decode frame.")
            continue

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # 이 프레임을 건너뛰고 다음 프레임으로 진행

        # 프레임 처리 로직을 여기에 계속 추가
        labels, cords = detector.score_frame(frame)
        original_w, original_h = frame.shape[1], frame.shape[0]
        detections = [(np.array(cord[:4].cpu()) * np.array([original_w, original_h, original_w, original_h]), cord[4].cpu().item()) for cord in cords]
        raw_detections = [(d[0], d[1], 'person') for d in detections]
        
        valid_detections = []
        for det in raw_detections:
            x1, y1, x2, y2 = det[0]
            if x1 < 0 or y1 < 0 or x2 > original_w or y2 > original_h or x2 <= x1 or y2 <= y1:
                continue
            crop = crop_image(frame, (x1, y1, x2, y2))
            if crop is None or crop.size == 0:
                continue
            valid_detections.append(det)

        tracks = []  # Always initialize tracks before using it
        if valid_detections:
            tracks = tracker.update_tracks(valid_detections, frame=frame)		

        for track in tracks:
            if track.is_confirmed() and (first_detected_id is None or track.track_id == first_detected_id):
                if first_detected_id is None:
                    first_detected_id = track.track_id
                bbox = track.to_tlbr()
                
                # 원래의 bbox: [x1, y1, x2, y2]
                # bbox의 가로 길이를 절반으로 줄임
                mid_x = (bbox[0] + bbox[2]) / 2
                half_width = (bbox[2] - bbox[0]) / 4  # 절반 길이의 절반을 계산
                x1_new = int(mid_x - half_width)
                x2_new = int(mid_x + half_width)
                
                # bbox 중심을 왼쪽으로 half_width 만큼 이동
                x1_new = int(mid_x - half_width - half_width)
                x2_new = int(mid_x + half_width - half_width)

                # 프레임 경계를 벗어나지 않도록 보정
                x1_new = max(0, min(x1_new, original_w - 1))
                x2_new = max(0, min(x2_new, original_w - 1))

                # 가로 길이 확인
                width = x2_new - x1_new
                if width > 0:
                    distance = 30000 / width
                else:
                    distance = float('inf')

                # 수정된 bbox 좌표로 사각형을 그림
                cv2.rectangle(frame, (x1_new, int(bbox[1])), (x2_new, int(bbox[3])), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {track.track_id}", (x1_new, int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                cv2.putText(frame, f"Distance: {distance:.2f} cm", (x1_new, int(bbox[1] - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                centroid = (int((x1_new + x2_new) / 2), int((bbox[1] + bbox[3]) / 2))
                cv2.arrowedLine(frame, (int(original_w/2), int(original_h/2)), (centroid[0], int(original_h/2)), (0, 150, 200), 2)

                # 객체 위치에 따라 직진, 우회전, 좌회전, 정지 수행
                if distance < 130:
                    send_direction_and_speed(command_conn, "stop", 0)
                    print("유모차가 정지합니다.")          # 실제에서는 라즈베리에 시그널 보내는 코드
                elif centroid[0] < frame.shape[1] / 2 - 50:    # 50픽셀 이상 벗어나는 걸 기준으로
                    send_direction_and_speed(command_conn, "left", 50)
                    print("유모차가 좌회전합니다.")
                elif centroid[0] > frame.shape[1] / 2 + 50:
                    send_direction_and_speed(command_conn, "right", 50)
                    print("유모차가 우회전합니다.")
                else:
                    send_direction_and_speed(command_conn, "forward", 100)
                    print("유모차가 직진합니다.")

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break

    client_socket.close()
    server_socket.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

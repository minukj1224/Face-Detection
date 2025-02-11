# # # 파일 경로 설정
# # txt_file = r"C:\Users\Minuk\Desktop\face_detection\test.txt"
# # output_txt_file = r"C:\Users\Minuk\Desktop\face_detection\test_cleaned.txt"

# # # 새로운 파일 저장
# # with open(txt_file, "r") as f, open(output_txt_file, "w") as out_f:
# #     for line in f:
# #         data = line.strip().split()
        
# #         # 이미지 파일명과 랜드마크 좌표만 남김 (바운딩 박스 제거)
# #         new_line = " ".join([data[0]] + data[5:])  # 파일명 + 랜드마크 좌표만
        
# #         out_f.write(new_line + "\n")

# # print(f"바운딩 박스 제거된 파일이 저장됨: {output_txt_file}")


# # import cv2
# # import os

# # # 파일 경로 설정
# # txt_file = r"C:\Users\Minuk\Desktop\face_detection\test_cleaned.txt"
# # image_folder = r"C:\Users\Minuk\Desktop\face_detection\test"
# # output_folder = r"C:\Users\Minuk\Desktop\face_detection\output_landmarks"

# # # 출력 폴더 생성
# # os.makedirs(output_folder, exist_ok=True)

# # # test_cleaned.txt 파일 읽기
# # with open(txt_file, "r") as f:
# #     lines = f.readlines()

# # for line in lines:
# #     data = line.strip().split()
    
# #     # 이미지 파일명
# #     img_name = data[0]
# #     img_path = os.path.join(image_folder, img_name)

# #     # 랜드마크 좌표
# #     landmarks = list(map(float, data[1:]))

# #     # 이미지 로드
# #     img = cv2.imread(img_path)
# #     if img is None:
# #         print(f"이미지를 찾을 수 없음: {img_path}")
# #         continue

# #     # 랜드마크 그리기 (빨간 점)
# #     for i in range(0, len(landmarks), 2):
# #         x, y = int(landmarks[i]), int(landmarks[i + 1])
# #         cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

# #     # 결과 이미지 저장
# #     save_path = os.path.join(output_folder, os.path.basename(img_name))
# #     cv2.imwrite(save_path, img)

# #     print(f"처리 완료: {save_path}")

# # print("모든 이미지 처리가 완료되었습니다.")


# # import cv2
# # import os

# # # 파일 경로 설정
# # txt_file = r"C:\Users\Minuk\Desktop\face_detection\test_cleaned.txt"
# # image_folder = r"C:\Users\Minuk\Desktop\face_detection\test"
# # output_folder = r"C:\Users\Minuk\Desktop\face_detection\output_resized_landmarks"
# # output_txt_file = r"C:\Users\Minuk\Desktop\face_detection\test_resized.txt"

# # # 리사이즈 크기
# # resize_w, resize_h = 96, 96

# # # 출력 폴더 생성
# # os.makedirs(output_folder, exist_ok=True)

# # # 변환된 랜드마크 좌표를 저장할 파일
# # with open(output_txt_file, "w") as out_f:
# #     with open(txt_file, "r") as f:
# #         lines = f.readlines()

# #     for line in lines:
# #         data = line.strip().split()
        
# #         # 이미지 파일명
# #         img_name = data[0]
# #         img_path = os.path.join(image_folder, img_name)

# #         # 랜드마크 좌표
# #         landmarks = list(map(float, data[1:]))

# #         # 이미지 로드
# #         img = cv2.imread(img_path)
# #         if img is None:
# #             print(f"이미지를 찾을 수 없음: {img_path}")
# #             continue

# #         # 원본 이미지 크기 가져오기
# #         orig_h, orig_w = img.shape[:2]

# #         # 이미지 리사이즈
# #         img_resized = cv2.resize(img, (resize_w, resize_h))

# #         # 랜드마크 좌표 변환 (비율 적용)
# #         scale_x = resize_w / orig_w
# #         scale_y = resize_h / orig_h
# #         resized_landmarks = []
# #         for i in range(0, len(landmarks), 2):
# #             new_x = int(landmarks[i] * scale_x)
# #             new_y = int(landmarks[i + 1] * scale_y)
# #             resized_landmarks.append(new_x)
# #             resized_landmarks.append(new_y)

# #         # 랜드마크 그리기 (빨간 점)
# #         for i in range(0, len(resized_landmarks), 2):
# #             x, y = resized_landmarks[i], resized_landmarks[i + 1]
# #             cv2.circle(img_resized, (x, y), 2, (0, 0, 255), -1)

# #         # 결과 이미지 저장
# #         save_path = os.path.join(output_folder, os.path.basename(img_name))
# #         cv2.imwrite(save_path, img_resized)

# #         # 변환된 좌표를 새로운 txt 파일에 저장
# #         new_line = " ".join([img_name] + list(map(str, resized_landmarks)))
# #         out_f.write(new_line + "\n")

# #         print(f"처리 완료: {save_path}")

# # print(f"모든 이미지 및 변환된 좌표가 저장되었습니다: {output_folder}, {output_txt_file}")


# import cv2
# import os

# # 원본 및 저장할 경로 설정
# train_txt_file = r"C:\Users\Minuk\Desktop\train\trainImageList.txt"
# test_txt_file = r"C:\Users\Minuk\Desktop\train\testImageList.txt"
# image_root = r"C:\Users\Minuk\Desktop\train"
# output_root = r"C:\Users\Minuk\Desktop\train_resized"
# target_size = (96, 96)  # 리사이즈 크기

# # 리사이즈된 랜드마크 파일 경로
# train_output_txt = os.path.join(output_root, "train.txt")
# test_output_txt = os.path.join(output_root, "test.txt")

# # 출력 폴더 생성
# os.makedirs(output_root, exist_ok=True)

# def process_file(txt_file, output_txt_file):
#     with open(txt_file, "r") as f, open(output_txt_file, "w") as out_f:
#         lines = f.readlines()

#         for line in lines:
#             data = line.strip().split()

#             # 이미지 파일 경로
#             img_path = os.path.join(image_root, data[0])
#             output_img_path = os.path.join(output_root, data[0])

#             # 원본 이미지 로드
#             img = cv2.imread(img_path)
#             if img is None:
#                 print(f"이미지를 찾을 수 없음: {img_path}")
#                 continue

#             # 원본 크기 가져오기
#             orig_h, orig_w = img.shape[:2]

#             # 이미지 리사이즈
#             img_resized = cv2.resize(img, target_size)

#             # 리사이즈된 이미지 저장
#             os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
#             cv2.imwrite(output_img_path, img_resized)

#             # 랜드마크 좌표 변환
#             landmarks = list(map(float, data[1:]))  # 바운딩 박스 포함 안 되어 있음
#             resized_landmarks = []
#             for i in range(0, len(landmarks), 2):
#                 x, y = landmarks[i], landmarks[i + 1]
#                 x_resized = (x / orig_w) * target_size[0]
#                 y_resized = (y / orig_h) * target_size[1]
#                 resized_landmarks.extend([x_resized, y_resized])

#             # 변환된 랜드마크를 새로운 파일에 저장
#             new_line = " ".join([data[0]] + [f"{x:.3f}" for x in resized_landmarks])
#             out_f.write(new_line + "\n")

#             print(f"리사이즈 및 저장 완료: {output_img_path}")

# # train 및 test 데이터 처리
# process_file(train_txt_file, train_output_txt)
# process_file(test_txt_file, test_output_txt)

# print("모든 이미지 리사이즈 및 랜드마크 변환이 완료되었습니다.")

import os
import torch
import cv2
import config
import numpy as np
from model import LandmarkModel

device = config.DEVICE
model = LandmarkModel().to(device)
model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
model.eval()

test_images = [os.path.join(config.DATASET_PATH, "test", f) for f in os.listdir(config.DATASET_PATH + "/test")]

for img_path in test_images:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, config.IMG_SIZE).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_resized).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor).cpu().numpy().reshape(-1, 2)

    for (x, y) in preds:
        cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

    cv2.imwrite(config.RANDOM_IMAGE_PATH + "/" + os.path.basename(img_path), img)

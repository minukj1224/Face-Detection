import cv2
import os

train_txt_file = r"./trainImageList.txt"
test_txt_file = r"./testImageList.txt"
image_root = r"./train"
output_root = r"./train_resized"
target_size = (96, 96)

train_output_txt = os.path.join(output_root, "train.txt")
test_output_txt = os.path.join(output_root, "test.txt")

os.makedirs(output_root, exist_ok=True)

def process_file(txt_file, output_txt_file):
    with open(txt_file, "r") as f, open(output_txt_file, "w") as out_f:
        lines = f.readlines()

        for line in lines:
            data = line.strip().split()

            img_path = os.path.join(image_root, data[0])
            output_img_path = os.path.join(output_root, data[0])

            img = cv2.imread(img_path)
            if img is None:
                print(f"이미지를 찾을 수 없음: {img_path}")
                continue

            orig_h, orig_w = img.shape[:2]

            img_resized = cv2.resize(img, target_size)

            os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
            cv2.imwrite(output_img_path, img_resized)

            landmarks = list(map(float, data[5:]))
            resized_landmarks = []
            for i in range(0, len(landmarks), 2):
                x, y = landmarks[i], landmarks[i + 1]
                x_resized = (x / orig_w) * target_size[0]
                y_resized = (y / orig_h) * target_size[1]
                resized_landmarks.extend([x_resized, y_resized])

            new_line = " ".join([data[0]] + [f"{x:.3f}" for x in resized_landmarks])
            out_f.write(new_line + "\n")

            print(f"리사이즈 및 저장 완료: {output_img_path}")

process_file(train_txt_file, train_output_txt)
process_file(test_txt_file, test_output_txt)

print("모든 이미지 리사이즈 및 랜드마크 변환이 완료되었습니다.")
import cv2
import os


x, y = 1418, 21
x2, y2 = 1702, 95

base_path = "/mnt/z/DATASETS/CoalFullImagesC&DBM/coal2026_Full_Images"
new_path = "/mnt/z/DATASETS/CoalFullImagesC&DBM/modifidHimanshuFiles"
os.makedirs(new_path ,exist_ok=True)

all_folders = os.listdir(base_path)

for i in all_folders:
    os.makedirs(os.path.join(new_path, i), exist_ok=True)
    all_images = os.listdir(os.path.join(base_path, i))
    for image in all_images:
        if image.endswith(".jpg"):
            img = cv2.imread(os.path.join(base_path, i , image))
            cv2.rectangle(img, (x, y), (x2, y2), (0,0,0), -1)
            cv2.imwrite(os.path.join(new_path, i, image), img)

# print(all_folders)

# import sys
# sys.exit(0)
# cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 0), -1)
# cv2.imwrite("./funny.jpg", image)


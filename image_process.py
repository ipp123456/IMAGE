import cv2
import numpy as np
import os

def get_image(path):       #获取图片
    img=cv2.imread(path)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img, gray

def Gaussian_Blur(gray):  # 高斯去噪
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    return blurred

def get_file_name(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = root+'\\'+file
            # if file_path[-3:] == 'jpg':
            file_list.append([file_path,file[:-4]])
    return file_list

def cut_pic(img_path, file_name, save_path):
    original_img, gray = get_image(img_path)  #  灰度化
    gblur = cv2.GaussianBlur(gray, (5, 5), 0)   # 去噪
    canny = cv2.Canny(gblur, 150, 300)   # 边缘检测
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    sure = cv2.dilate(canny, kernel, iterations=1)
    (cnts, _) = cv2.findContours(sure.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 分割
    coins = original_img.copy()
    cv2.drawContours(coins, cnts, -1, (0, 255, 0), 2)
    epsilon = 0.001
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        coin = gray[y:y + h, x:x + w]

        pts = c - c.min(axis=0)
        mask = np.zeros(coin.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        dst = cv2.bitwise_and(coin, coin, mask=mask)
        bg = np.ones_like(coin, np.uint8) * 255
        cv2.bitwise_not(bg, bg, mask=mask)
        dst2 = bg + dst

        (_, thresh) = cv2.threshold(dst2, 160, 255, cv2.THRESH_BINARY)  # 二值化
        save_img = cv2.resize(thresh, (300, 200), )

        cv2.imwrite(save_path+'\\'+file_name+'-obj-'+str(i)+'.jpg', save_img)


file_list = get_file_name('data')
save_path = 'save'
for file_path, file_name in file_list:
    cut_pic(file_path, file_name, save_path)





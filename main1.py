import cv2
import numpy as np


# 可视化
def img_show(name, img):
    cv2.namedWindow(name, 0)
    cv2.resizeWindow(name, 1000, 500)
    cv2.imshow(name, img)
    cv2.waitKey(0)


def color_warped(path):
    img = cv2.imread(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 颜色识别(红色)，过滤红色区域
    lower_red1 = np.array([0, 43, 46])  # 红色阈值下界
    higher_red1 = np.array([10, 255, 255])  # 红色阈值上界
    mask_red1 = cv2.inRange(hsv, lower_red1, higher_red1)
    lower_red2 = np.array([156, 43, 46])  # 红色阈值下界
    higher_red2 = np.array([180, 255, 255])  # 红色阈值上界
    mask_red2 = cv2.inRange(hsv, lower_red2, higher_red2)
    mask_red = cv2.add(mask_red1, mask_red2)  # 拼接过滤后的mask
    img_show('mask_red', mask_red)

    # 形态学去噪，cv2.MORPH_OPEN先腐蚀再膨胀，cv2.MORPH_CLOSE先膨胀再腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=3)
    img_show('mask_red', mask_red)

    # 轮廓检测，找出线条的轮廓
    draw_cnt = img.copy()
    cnts = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    cv2.drawContours(draw_cnt, cnts, -1, (0, 255, 0), 2)
    img_show('draw_cnt', draw_cnt)

    # 四边形拟合，找到相应的的顶点
    draw_approx = img.copy()
    point1, point2 = list(), list()
    for cnt in cnts:
        for epsilon in range(50):
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4:
                break
        cv2.polylines(draw_approx, [approx], True, (0, 0, 255), 2)  # 绘制边
        for i in approx:
            cv2.circle(draw_approx, i[0], 6, (0, 0, 0), -1)  # 绘制顶点

        approx = [i[0] for i in approx.tolist()]
        approx = sorted(approx, key=lambda k: k[1], reverse=False)  # 按y坐标排序，升序

        point1.extend(approx[:2])  # 存放上顶点坐标
        point2.extend(approx[2:])  # 存放下顶点坐标
    point1.sort(key=lambda k: k[0], reverse=False)  # 按x坐标排序，升序
    point2.sort(key=lambda k: k[0], reverse=False)
    img_show('draw_approx', draw_approx)

    # 透视矫正目标区域
    w, h = 900, 300
    rect = [point1[0], point1[-1], point2[-1], point2[0]]  # 顺序为第一个四边形的左上，第四个四边形的右上，第四个四边形的右下，第一个四边形的左下
    pts1 = np.array(rect, dtype="float32")
    pts2 = np.array([rect[0], [rect[0][0] + w, rect[0][1]],
                     [rect[0][0] + w, rect[0][1] + h], [rect[0][0], rect[0][1] + h]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts1, pts2)  # 变换矩阵
    img_warped = cv2.warpPerspective(img, M, (1500, 500))  # 透视变换
    img_show('img_warped1', img_warped)

    img_warped = img_warped[rect[0][1]: rect[0][1] + h, rect[0][0]: rect[0][0] + w]  # 抠出变换后的区域
    img_show('img_warped2', img_warped)


if __name__ == '__main__':
    path = './1.jpg'
    color_warped(path)
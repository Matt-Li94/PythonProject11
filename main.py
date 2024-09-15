import cv2
import numpy as np

if __name__ == "__main__":
    img = cv2.imread('./3.jpg')
    # 在彩色图像的情况下，解码图像将以b g r顺序存储通道。
    grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 从RGB色彩空间转换到HSV色彩空间
    grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)

    # H、S、V范围一：
    lower1 = np.array([0, 43, 46])
    upper1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(grid_HSV, lower1, upper1)  # mask1 为二值图像
    res1 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask1)

    # H、S、V范围二：
    lower2 = np.array([156, 43, 46])
    upper2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(grid_HSV, lower2, upper2)
    res2 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask2)

    # 将两个二值图像结果 相加
    mask3 = mask1 + mask2

    # 结果显示
    cv2.imshow("mask3", mask3)
    cv2.imshow("img", img)
    # cv2.imshow("Mask1", mask1)
    # cv2.imshow("res1", res1)
    # cv2.imshow("Mask2", mask2)
    # cv2.imshow("res2", res2)
    # cv2.imshow("grid_RGB", grid_RGB[:, :, ::-1])  # imshow()函数传入的变量也要为b g r通道顺序
    cv2.waitKey(0)
    cv2.destroyAllWindows()


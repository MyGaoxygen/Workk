import cv2
import numpy as np

cornerLocationPoint = list()
treasureXY = list()

cap = cv2.VideoCapture(0)


def findTreasureXY():
    while cap.isOpened():
        # 逐帧读取视频
        ret, frame = cap.read()

        if ret:
            # 在这里对每一帧进行处理
            # 将图像转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 进行边缘检测
            edges = cv2.Canny(gray, 200, 300)

            # 轮廓提取
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cornerLocationPoint.clear()
            # 遍历轮廓
            for contour in contours:
                # 计算轮廓的边界框
                center, WH, _ = cv2.minAreaRect(contour)  # 找最小外接矩形
                w = WH[0]
                h = WH[1]
                if w > 40 or h > 40 or w < 20 or h < 20:  # 过大过小直接过滤
                    continue

                # 判断边界框是否接近正方形
                if abs(w - h) < 5:
                    # 绘制边界框
                    x = center[0]
                    y = center[1]
                    cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                  (0, 255, 0), 2)
                    cornerLocationPoint.append([x, y, w, h])

            # 显示帧
            cv2.imshow('Camera', frame)

            # 按下 'Q' 键推出
            if cv2.waitKey(1) == ord('Q') or cv2.waitKey(1) == ord('q'):
                # 释放资源
                cap.release()
                cv2.destroyAllWindows()
                break


            # 当检测到四个像素点时进行截图和透视变换
            if len(cornerLocationPoint) == 4:
            # 排序
                cornerLocationPoint.sort(key=lambda member: member[0], reverse=False)
                cache = list()
                if cornerLocationPoint[0][1] < cornerLocationPoint[1][1]:
                    leftTop = cornerLocationPoint.pop(0)
                    leftBottom = cornerLocationPoint.pop(0)
                    cache.append(leftTop)
                else:
                    leftTop = cornerLocationPoint.pop(1)
                    leftBottom = cornerLocationPoint.pop(0)
                    cache.append(leftTop)
                    cornerLocationPoint.sort(key=lambda member: member[1], reverse=False)
                    rightTop = cornerLocationPoint.pop(0)
                    rightBottom = cornerLocationPoint.pop(0)
                    cache.append(rightTop)
                    cache.append(leftBottom)
                    cache.append(rightBottom)

                    X = [cache[0][0], cache[1][0], cache[2][0], cache[3][0]]
                    Y = [cache[0][1], cache[1][1], cache[2][1], cache[3][1]]

                    # 透视变换
                    pic1 = np.float32([[X[0], Y[0]], [X[1], Y[1]], [X[2], Y[2]], [X[3], Y[3]]])
                    pic2 = np.float32([[0, 0], [640, 0], [0, 640], [640, 640]])
                    Matrix = cv2.getPerspectiveTransform(pic1, pic2)
                    perspectivePic = cv2.warpPerspective(frame, Matrix, (640, 640))
                    # cv2.imshow("PerspectiveTransform", perspectivePic)
                    # 截取迷宫区域
                    cropped_image = perspectivePic[int(cache[0][2] * 2.2):int(640 - cache[1][2] * 2.2),
                                    int(0 + cache[0][3] * 2.2):int(640 - cache[2][3] * 2.2)]

                    # 将彩色图像转换为灰度图像
                    perspectivePic = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    # # 应用阈值处理，将灰度图像二值化为黑白图像
                    _, binary_image = cv2.threshold(perspectivePic, 108, 255, cv2.THRESH_BINARY)  # 光照调整
                    # cv2.imshow("THRESH_BINARY", binary_image)

                    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    treasureXY.clear()

                    w, h = perspectivePic.shape
                    unitW = int(w / 10)
                    unitH = int(h / 10)

                for contour in contours:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    area = cv2.contourArea(contour)
                    if 5 < radius < 12:
                        center = (int(x), int(y))
                        radius = int(radius)
                        cv2.circle(perspectivePic, center, radius, (0, 0, 255), 5)

                        x = int(x / unitW) * 2 + 1
                        y = int(y / unitH) * 2 + 1
                        treasureXY.append([int(y), int(x)])
                        cv2.imshow('DONE', perspectivePic)

                        print("识别出的宝藏个数：", len(treasureXY))
                        print("识别出的宝藏坐标：", treasureXY)
                        cv2.waitKey(0)
                        # 释放资源
                        cv2.destroyAllWindows()
                        return treasureXY

        else:
            print("摄像头ret有问题")
            break

            findTreasureXY()
import cv2
import numpy as np
import cv2.aruco as aruco

cam = cv2.VideoCapture(0)
img_counter = 0
cv2.namedWindow("cam")
marker_size = 8
num_aruco = 50
qrCodeDetector = cv2.QRCodeDetector()
# camera_matrix =
# dist_coeff =


def get_bin_img(frame: np.ndarray):
    # print(type(frame))h
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th, img = cv2.threshold(grayframe, 0, 255, cv2.THRESH_OTSU)
    return img


def get_aspect_ratio(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    return aspect_ratio


while True:
    # grab frame, threshold it, identify qr codes
    ret, frame = cam.read()
    otsu_frame = get_bin_img(frame)

    # rectangular kernal to close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closing = cv2.morphologyEx(otsu_frame, cv2.MORPH_CLOSE, kernel)

    # invert the image
    inv_img = cv2.bitwise_not(closing)
    # find contoursh
    cnt, hier = cv2.findContours(inv_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # filter contours by area
    cnt_filtered = [
        c
        for c in cnt
        if (500 < cv2.contourArea(c) < 100000) & (0.7 < get_aspect_ratio(c) < 1.2)
    ]

    #
    disp_img = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)

    # (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
    x, y, w, h = cv2.boundingRect(inv_img)
    cv2.rectangle(disp_img, (x, y), (x + w, y + h), (255, 0, 0), 3)

    # txt, points, straight_qrcode = qrCodeDetector.detectAndDecode(disp_img)

    key_list = [
        "DICT_6X6_100",
        "DICT_6X6_250",
        "DICT_6X6_1000",
        "DICT_ARUCO_ORIGINAL",
        "DICT_APRILTAG_36h10",
        "DICT_APRILTAG_36h11",
    ]

    for dict_key in key_list:
        key = getattr(aruco, dict_key)
        aruco_dict = aruco.Dictionary_get(key)
        aruco_params = aruco.DetectorParameters_create()
        corners, ids, rejected = aruco.detectMarkers(
            disp_img, aruco_dict, parameters=aruco_params
        )

        # Check that at least one ArUco marker was detected
        if ids is not None:
            for id in ids:
                print(id)
                print(dict_key)

            # Draw a square around detected markers in the video frame
            aruco_img = aruco.drawDetectedMarkers(disp_img, corners, ids, (255, 255, 0))

            # Get the rotation and translation vectors
            # rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
            #     corners, 0.1, camera_matrix, dist_coeff
            # )

            # display the result
            cv2.imshow("cam", aruco_img)
        else:
            cv2.imshow("cam", disp_img)
    # if points is not None:b
    #     print(txt)

    # if points is not None:
    #     num_points = len(points)
    #     for i in range(num_points):
    #         n_point_idx = (i + 1) % num_points
    #         cv2.line(
    #             disp_img,
    #             tuple(points[n_point_idx][0]),
    #             tuple(points[n_point_idx][1]),
    #             (255, 0, 0),
    #             5,
    #         ),

    # draw contours
    # cnt_img = cv2.drawContours(disp_img, cnt_filtered, -1, (0, 0, 255), 3)

    # draw

    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, aruco_img)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()

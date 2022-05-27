import cv2


def get_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    gray[thresh == 255] = 0

    _, threshold = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    cnts, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return cnts


if __name__ == "__main__":
    img = cv2.imread('../dataset/images/delta_perm8/854.png')

    print(get_contours(img))

    for contour in get_contours(img):

        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)

        cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)

        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])

        cv2.imshow('shapes', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()



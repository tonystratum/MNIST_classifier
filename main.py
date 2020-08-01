import cv2
import numpy as np

from PIL import Image

from net import predict

drawing = False  # true if mouse is pressed
mode = False  # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1
color = (255, 255, 255)
pen_size = 25
cache_path = 'cache'


# mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode, color, cache_path, pen_size

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv2.rectangle(img, (ix, iy), (x, y), color, -1)
            else:
                cv2.circle(img, (x, y), pen_size, color, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv2.rectangle(img, (ix, iy), (x, y), color, -1)
        else:
            cv2.circle(img, (x, y), pen_size, color, -1)

        # this code is executed
        # after user drew something

        # TODO: reshape (512, 512, 3) to (28, 28)
        # divide img into 3 channels (NO NEED)
        # b, g, r = map(np.squeeze, np.split(img, 3, axis=2))

        # make a single array shape = (512, 512)
        # avg of all channels (grey)
        grey = np.mean(img, axis=2)
        #print(grey)

        # TODO: downsample it WITH NUMPY
        # pil for now
        grey_ds = np.asarray(Image.fromarray(grey).resize(size=(28, 28), resample=Image.LANCZOS)) / 255
        #print(grey_ds)

        # reshape to (784, 1) vector
        grey_ds = grey_ds.reshape(-1, 1)

        # predict
        predictions = predict(grey_ds, cache_path).reshape(-1)
        print(predictions * 100)
        print(np.argmax(predictions))


img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == ord('d'):
        img = np.zeros((512, 512, 3), np.uint8)
    elif k == 27:
        break

cv2.destroyAllWindows()

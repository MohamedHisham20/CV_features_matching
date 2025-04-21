import logging
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

logger = logging.getLogger(__name__)

def extract_corners(image, threshold=0.95, using="harris"):
    start_time = cv2.getTickCount()
    # make sure image is grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    I_x, I_y = computeGradients(image)

    I_xx = gaussian_filter(I_x * I_x, sigma=1)
    I_yy = gaussian_filter(I_y * I_y, sigma=1)
    I_xy = gaussian_filter(I_x * I_y, sigma=1)

    corners = detectCorneres(I_xx, I_yy, I_xy, using)

    # Thresholding
    # normalize values to [0, 1]
    corners = corners - np.min(corners)
    corners = corners / np.max(corners)
    mask = corners > threshold

    # non-max suppression
    corners = non_max_suppression(mask, corners)

    end_time = cv2.getTickCount()
    time = (end_time - start_time) / cv2.getTickFrequency()

    return corners, time


def computeGradients(image):
    # Sobel kernels
    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Compute gradients
    I_x = cv2.filter2D(image, -1, kernel_x)
    I_y = cv2.filter2D(image, -1, kernel_y)

    return I_x, I_y


def detectCorneres(I_xx, I_yy, I_xy, using):
    if using != "harris" and using != "lambda":
        raise ValueError("Using must be either 'harris' or 'lambda'")

    harris = None
    lambda_negative = None

    if using == "harris":
        harris = np.zeros(I_xx.shape, dtype=np.float32)
    elif using == "lambda":
        lambda_negative = np.zeros(I_xx.shape, dtype=np.float32)

    for i in range(0, I_xx.shape[0]):
        for j in range(0, I_xx.shape[1]):
            S_xx = I_xx[i, j]
            S_yy = I_yy[i, j]
            S_xy = I_xy[i, j]

            if using == "harris":
                det = S_xx * S_yy - S_xy * S_xy
                trace = S_xx + S_yy
                k = 0.04
                harris[i, j] = det - k * trace * trace
            elif using == "lambda":
                eigen_values, eigen_vectors = np.linalg.eig(np.array([[S_xx, S_xy], [S_xy, S_yy]]))
                idx = eigen_values.argsort()[::-1]
                eigen_values = eigen_values[idx]
                lambda_negative[i, j] = eigen_values[1]

    if using == "harris":
        return harris
    elif using == "lambda":
        return lambda_negative


def non_max_suppression(mask, corners):
    # Pad both the corners and mask arrays to handle edges
    padded_corners = np.pad(corners, pad_width=1, mode='constant', constant_values=0)
    padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)

    coords = np.argwhere(mask)
    for coord in coords:
        x, y = coord
        # Shift coordinates because of padding
        x_p, y_p = x + 1, y + 1
        window = padded_corners[x_p - 1:x_p + 2, y_p - 1:y_p + 2]
        if padded_corners[x_p, y_p] < np.max(window):
            padded_mask[x_p, y_p] = 0
        else:
            padded_mask[x_p, y_p] = 1

    return padded_mask[1:-1, 1:-1]


if __name__ == "__main__":
    # Example usage
    image = cv2.imread('Test Images/puppy.jpg')
    corners = extract_corners(image, threshold=0.9, using="lambda")

    image_with_corners = image.copy()

    corner_coords = np.argwhere(corners)  # returns (y, x)

    for y, x in corner_coords:
        cv2.circle(image_with_corners, (x, y), radius=2, color=(255, 0, 0), thickness=-1)  # BGR: Blue

    cv2.imshow('Corners', image_with_corners)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

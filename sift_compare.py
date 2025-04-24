import cv2 as cv
import numpy as np

from SIFT import extractKeyPointsandDescriptor


def test():
    # Load test image
    image = cv.imread('feature_imgs/poaaaaa.jpg', cv.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found!")
        return

    print(f"[DEBUG] Loaded test image with shape {image.shape}")

    # start time calculations
    start_time = cv.getTickCount()
    print(f"[DEBUG] Starting SIFT computation...{start_time}")
    # Run custom SIFT implementation
    keypoints, descriptors = extractKeyPointsandDescriptor(image)
    end_time = cv.getTickCount()
    time_taken = (end_time - start_time) / cv.getTickFrequency()
    print(f"[DEBUG] SIFT computation completed in {time_taken:.4f} seconds")
    print(f"[DEBUG] Number of keypoints detected: {len(keypoints)}")
    print(f"[DEBUG] Descriptor shape: {descriptors.shape}")


    # Draw custom keypoints
    img_custom = image.copy()
    img_custom = cv.drawKeypoints(image, keypoints, img_custom,
                                  flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Run OpenCV SIFT
    sift_built_in = cv.SIFT_create()
    kp_opencv, desc_opencv = sift_built_in.detectAndCompute(image, None)

    # Draw OpenCV keypoints
    img_opencv = image.copy()
    img_opencv = cv.drawKeypoints(image, kp_opencv, img_opencv,
                                  flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Print comparison stats
    print("\n=== Comparison Results ===")
    print(f"Custom SIFT keypoints: {len(keypoints)}")
    print(f"OpenCV SIFT keypoints: {len(kp_opencv)}")
    print(f"Keypoint ratio (custom/OpenCV): {len(keypoints) / len(kp_opencv):.2f}")

    if len(keypoints) > 0:
        print(f"Custom descriptor shape: {descriptors.shape}")
    print(f"OpenCV descriptor shape: {desc_opencv.shape}")

    # Display results side by side
    comparison = np.hstack((img_custom, img_opencv))
    cv.imshow("SIFT Comparison: Custom (Left) vs OpenCV (Right)", comparison)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Save results
    cv.imwrite("custom_sift.jpg", img_custom)
    cv.imwrite("opencv_sift.jpg", img_opencv)
    cv.imwrite("sift_comparison.jpg", comparison)

test()

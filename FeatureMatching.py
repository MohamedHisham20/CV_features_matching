import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk

class Matching:
    
    def __init__(self, image1, keypoints1, descriptor1, image2, keypoints2, descriptor2):
        self.image1 = image1
        self.keypoints1 = keypoints1
        self.descriptor1 = descriptor1
        self.image2 = image2
        self.keypoints2 = keypoints2
        self.descriptor2 = descriptor2
        self.matches = []

    ###### built in sift bypass for now
    @staticmethod
    def extract_sift_features(image_path1, image_path2):
        image1 = cv2.imread(image_path1, cv2.IMREAD_COLOR)
        image2 = cv2.imread(image_path2, cv2.IMREAD_COLOR)

        if image1 is None or image2 is None:
            raise ValueError("One or both image paths are invalid or the images could not be loaded.")
        
        # Convert to grayscale 
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()

        keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

        print(f"Extracted {len(keypoints1)} keypoints from image 1")
        print(f"Extracted {len(keypoints2)} keypoints from image 2")

        return image1, keypoints1, descriptors1, image2, keypoints2, descriptors2

    def match_ssd(self, threshold=30000):
        matches = []
        for d1 in self.descriptor1:
                distances = []
                for d2 in self.descriptor2:
                    distance = np.sum((d1 - d2)**2)
                    distances.append(distance)
                min_distance = np.min(distances)
                best_match_index = np.argmin(distances)
                #add matches if under threshold
                if min_distance < threshold:
                    matches.append(best_match_index)
                else:
                    matches.append(None)  
        
        print(f"Number of good matches : {len(matches)}")
    
        return matches
    
    def match_ncc(self, threshold = 0.97):
        matches = []
        for d1 in self.descriptor1:
            correlations = []
            for d2 in self.descriptor2:
                correlation = np.sum(
                    d1 * d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
                correlations.append(correlation)
            max_correlation = np.max(correlations)
            best_match_index = np.argmax(correlations)
            #add matches if above threshold
            if max_correlation > threshold:
                matches.append(best_match_index)
            else:
                matches.append(None)
        return matches

    def visualize_matches(self, good_matches):
        h1, w1 = self.image1.shape
        h2, w2 = self.image2.shape
        canvas_height = max(h1, h2)
        canvas_width = w1 + w2
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        img1_color = cv2.cvtColor(self.image1, cv2.COLOR_GRAY2RGB)
        img2_color = cv2.cvtColor(self.image2, cv2.COLOR_GRAY2RGB)

        canvas[:h1, :w1, :] = img1_color
        canvas[:h2, w1:, :] = img2_color

        plt.figure(figsize=(15, 10))
        plt.imshow(canvas)
        
        for i, match_idx in enumerate(good_matches):
            if match_idx is not None:
                pt1 = tuple(map(int, self.keypoints1[i].pt))
                pt2 = tuple(map(int, self.keypoints2[match_idx].pt))
                pt2 = (int(pt2[0] + w1), int(pt2[1]))  
                color = np.random.rand(3,)  

                plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=0.75)
                plt.scatter(*pt1,  s=20, marker='o', color=color)
                plt.scatter(*pt2,  s=20, marker='o', color=color)

        plt.axis('off')
        plt.title('Feature Matches')
        plt.show()


#####for testing
def select_and_run():
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select Two Images",
        filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")]
    )

    if len(file_paths) != 2:
        print("Please select exactly two images.")
        return

    image1, keypoints1, descriptors1, image2, keypoints2, descriptors2 = Matching.extract_sift_features(file_paths[0], file_paths[1])

    matcher = Matching(image1, keypoints1, descriptors1, image2, keypoints2, descriptors2)

    good_matches = matcher.match_ssd()
    print(f"Number of good matches: {len(good_matches)}")
    matcher.visualize_matches(good_matches)

if __name__ == "__main__":
    select_and_run()
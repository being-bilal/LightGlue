import os
import time
import csv
import cv2
import numpy as np
from lightglue import SuperPoint, LightGlue
from lightglue.utils import load_image, rbd

# Initialize model (CPU, with adaptive pruning)
extractor = SuperPoint(max_num_keypoints=256).eval()
matcher = LightGlue(
    features='superpoint',
    depth_confidence=0.9,
    width_confidence=0.95
).eval()

# Paths and file loading
image_dir = 'LightGlue/dataset/images'
image_files = sorted(os.listdir(image_dir), key=lambda x: int(''.join(filter(str.isdigit, x))))
num_pairs = len(image_files) - 1

# Output folders
csv_path = 'matching_results.csv'
vis_dir = 'dataset/match_vis'
os.makedirs(vis_dir, exist_ok=True)

# Run matching and save results
total_time = 0
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['PairIndex', 'Image1', 'Image2', 'Time(s)', 'Matches'])

    for i in range(num_pairs):
        image1 = image_files[i]
        image2 = image_files[i + 1]

        img0_path = os.path.join(image_dir, image1)
        img1_path = os.path.join(image_dir, image2)

        # Load tensors for LightGlue
        image0 = load_image(img0_path)
        image1_tensor = load_image(img1_path)

        start_time = time.time()

        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1_tensor)
        matches_dict = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches_dict = [rbd(x) for x in [feats0, feats1, matches_dict]]
        matches = matches_dict['matches']

        end_time = time.time()
        elapsed = end_time - start_time
        total_time += elapsed

        # Log result
        print(f'Pair {i}: {elapsed:.3f} s, Matches: {matches.shape[0]} — {image1} vs {image2}')
        writer.writerow([i, image1, image2, f'{elapsed:.3f}', matches.shape[0]])

        # ---- Visualization ----
        pts0 = feats0['keypoints'][matches[:, 0]].cpu().numpy()
        pts1 = feats1['keypoints'][matches[:, 1]].cpu().numpy()

        # Load raw images with OpenCV
        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)

        # Resize img1 to match height of img0
        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]
        img1 = cv2.resize(img1, (int(w1 * h0 / h1), h0))

        # Combine images side by side
        vis_image = np.hstack((img0, img1))

        # Draw matches
        for (x0, y0), (x1, y1) in zip(pts0, pts1):
            pt1 = (int(x0), int(y0))
            pt2 = (int(x1 + img0.shape[1]), int(y1))  # offset x for second image
            cv2.line(vis_image, pt1, pt2, (0, 255, 0), 1)

        # Save visualization
        vis_path = os.path.join(vis_dir, f'match_{i:04d}.png')
        cv2.imwrite(vis_path, vis_image)

# Final summary
avg_time = total_time / num_pairs
print(f'\nAverage time per image pair: {avg_time:.3f} seconds')

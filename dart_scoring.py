#=============Config=============#

#Calibration
radius_calibration = 1.02
x_offset = 0
y_offset = -15

#File Export
file_export = True

#================================#

import sys
import os
import cv2 as cv
import numpy as np


def align_images(reference_img, input_img):
    # Convert images to grayscale
    gray_ref = cv.cvtColor(reference_img, cv.COLOR_BGR2GRAY)
    gray_input = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)

    # Use ORB to detect and compute features
    orb = cv.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray_ref, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_input, None)

    # Use BFMatcher to find the best matches between descriptors
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the matched keypoints
    ref_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    input_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography matrix
    H, _ = cv.findHomography(input_pts, ref_pts, cv.RANSAC)

    # Warp input image to align with reference image
    height, width, channels = reference_img.shape
    aligned_input_img = cv.warpPerspective(input_img, H, (width, height))

    return aligned_input_img


def draw_zones(image, center, radius):
    score_table = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    for i in range(20):
        #Determine zone score
        zone_number = score_table[i] 

        #Calculate zone borders
        start_angle = (i * 18) - 99  
        end_angle = start_angle + 18
        mid_angle = (start_angle + end_angle) / 2

        # Draw zone borders
        x1 = int(center[0] + radius * np.cos(np.radians(start_angle)))
        y1 = int(center[1] + radius * np.sin(np.radians(start_angle)))
        x2 = int(center[0] + radius * np.cos(np.radians(end_angle)))
        y2 = int(center[1] + radius * np.sin(np.radians(end_angle)))
        cv.line(image, center, (x1, y1), (0, 255, 0), 2)
        cv.line(image, center, (x2, y2), (0, 255, 0), 2)

        # Draw zone numbers
        label_x = int(center[0] + (radius * 0.6) * np.cos(np.radians(mid_angle))) #calculate label x coord
        label_y = int(center[1] + (radius * 0.6) * np.sin(np.radians(mid_angle))) #calculate label y coord
        cv.putText(image, str(zone_number), (label_x, label_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) #draw label
    
    # Draw modifier zone borders
    for scale in [0.42, 0.48, 0.7, 0.76]:
        cv.circle(image, center, int(radius * scale), (0, 255, 0), 2)


def main(argv):

    #==========Import Images========#

    default_reference = 'smarties.png'
    default_input = 'input.png'
    
    # Parse arguments
    verbose = False
    reference_filename = default_reference
    input_filename = default_input
    
    if len(argv) > 0:
        for arg in argv:
            if arg.lower() == "verbose=true":
                verbose = True
            elif arg.lower() == "verbose=false":
                verbose = False
            else:
                if reference_filename == default_reference:
                    reference_filename = arg
                else:
                    input_filename = arg
    
    # Loads the reference image
    reference = cv.imread(cv.samples.findFile(reference_filename), cv.IMREAD_COLOR)
    # Loads the input image
    input_img = cv.imread(cv.samples.findFile(input_filename), cv.IMREAD_COLOR)


    # Check if images are loaded fine
    if reference is None or input_img is None:
        print('Error opening image!')
        print('Usage: dart)scoring.py [reference_image_name -- default ' + default_reference + '] [input_image_name -- default ' + default_input + '] [verbose=true/false]\n')
        return -1

    #================================#


    #========Initialise points=======#

    total_score = 0
    score_array = []
    score_table = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

    #================================#

    #=======File Export Config=======#

    # Create Outputs directory if it doesn't exist
    output_dir = 'Outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #================================#


    #======Process Input Images======#

    # Align the input image to the reference image
    aligned_input_img = align_images(reference, input_img)

    ref_img = np.copy(reference)
    input_copy = np.copy(aligned_input_img)

    gray = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
    
    # Boost contrast with histogram equalization
    gray_equalized = cv.equalizeHist(gray)

    # Apply median blur
    gray_blurred = cv.medianBlur(gray_equalized, 5)
    
    blurred = cv.medianBlur(gray_blurred, 151)
    rows = blurred.shape[0]
    
    circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, 1, rows / 8,
                              param1=100, param2=30,
                              minRadius=100, maxRadius=1000)
    
    #================================#

    #=========Detect Circles=========#

    mask = None  # Initialize mask variable
    circle_center = None

    if circles is not None:
        circles = np.uint16(np.around(circles))
        largest_circle = circles[0][0]  # Select the first detected circle as the largest
        circle_center = (largest_circle[0] - x_offset, largest_circle[1] - y_offset)  # Get the center of the largest circle
        mask = np.zeros_like(blurred)
        # Draw filled circle on mask
        cv.circle(mask, circle_center, int(largest_circle[2]), (255, 255, 255), -1)

        # Draw the largest circle on the blurred image
        blurred_with_circle = blurred.copy()
        cv.circle(blurred_with_circle, circle_center, int(largest_circle[2]), (0, 255, 0), 3)
        if file_export:
            cv.imwrite(os.path.join(output_dir, "Largest_Circle_on_Blurred_Image.png"), blurred_with_circle)

        if verbose:
            cv.imshow("Largest Circle on Blurred Image", blurred_with_circle)

    #================================#

    #==========Extract Board=========#

    # Set everything outside the circles to black
    if mask is not None:
        ref_masked = np.zeros_like(ref_img)
        input_masked = np.zeros_like(input_copy)

        ref_masked[mask != 0] = ref_img[mask != 0]
        input_masked[mask != 0] = input_copy[mask != 0]

        # Apply increased Gaussian blur to both masked images to reduce noise
        ref_blurred = cv.GaussianBlur(ref_masked, (31, 31), 0)
        input_blurred = cv.GaussianBlur(input_masked, (31, 31), 0)

        difference = cv.absdiff(input_blurred, ref_blurred) #find darts as difference between reference and input
        difference_gray = cv.cvtColor(difference, cv.COLOR_BGR2GRAY) #convert difference to grayscale for thresholding
        _, diff_thresh = cv.threshold(difference_gray, 10, 255, cv.THRESH_BINARY) #threshold difference

        if file_export:
            cv.imwrite(os.path.join(output_dir, "detected_circles.png"), mask)
            cv.imwrite(os.path.join(output_dir, "blurred.png"), blurred)

        if verbose:
            cv.imshow("detected circles", mask)
            cv.imshow("blurred", blurred)
    else:
        print("No circles detected.")
        if verbose:
            cv.imshow("blurred", blurred)

    #================================#

    #==========Locate Darts==========#

    if circle_center is not None:
        circle_radius = circles[0][0][2]*radius_calibration

        # Find all blobs in the thresholded image
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(diff_thresh, connectivity=8)

        # Sort blobs by area (the third column of stats)
        blob_areas = stats[1:, cv.CC_STAT_AREA]  # Ignore the background label (0)
        largest_blobs_indices = np.argsort(blob_areas)[-3:][::-1] + 1  # Get indices of the three largest blobs

        polar_coordinates = [] #initialise polar coords array

        cv.drawMarker(input_copy, circle_center, (0, 255, 0), cv.MARKER_CROSS, markerSize=20, thickness=3) #mark board centre

        for i in largest_blobs_indices: #for the detected darts
            dart_center = (int(centroids[i][0]), int(centroids[i][1]))
            dx = dart_center[0] - circle_center[0] #calculate x distance from centre
            dy = dart_center[1] - circle_center[1] #calculate y distance from centre
            radius = np.sqrt(dx**2 + dy**2) / circle_radius  # Normalize radius
            theta = np.arctan2(dy, dx)  # Angle relative to circle's center
            polar_coordinates.append((radius, theta, dart_center)) #store dart coordinates

            #mark dart on dart board
            cv.drawMarker(input_copy, dart_center, (0, 255, 0), cv.MARKER_CROSS, markerSize=20, thickness=3)

        # Determine the zone for each dart
        for radius, theta, dart_center in polar_coordinates:
            
            theta = np.degrees(theta) + 90  # Adjust angle to be relative to the top
            if theta < 0:
                theta += 360  # Ensure positive angle
            # Convert angle to a zone number
            zone = (int(theta + 9) % 360 // 18) + 1  # Shift the angle by 9 degrees to center zone 1 at the top

            if 0.42 < radius < 0.48:
                dart_score = 3 * score_table[zone - 1]
            elif 0.7 < radius < 0.76:
                dart_score = 2 * score_table[zone - 1]
            elif radius >= 0.76:
                dart_score = 0
            else:
                dart_score = score_table[zone - 1]
            total_score += dart_score
            score_array.append(dart_score)
            print(f"Dart center: {dart_center}, Normalized Radius: {radius}, Angle (degrees): {theta}, Score: {dart_score}")

    score_array.sort(reverse=True)
    score_array += [0] * (3 - len(score_array))  # Pad with zeros if less than 3 blobs

    #================================#

    #======Highlight Differences=====#

    # Convert diff_thresh to a 3-channel image
    diff_thresh_color = cv.cvtColor(diff_thresh, cv.COLOR_GRAY2BGR)
    # Highlight differences in red
    highlighted_diff = cv.addWeighted(input_copy, 0.7, diff_thresh_color, 0.3, 0)

    #================================#

    #===========Mark Output==========#

    # Draw zones on a copy of the input image
    input_with_zones = np.copy(input_copy)
    if circle_center is not None:
        draw_zones(input_with_zones, circle_center, circle_radius)

    #================================#

    #==========Display Score=========#

    final_with_score = input_with_zones.copy()
    # Display the scores
    score_text = [
        f"dart 1 score: {score_array[0]}",
        f"dart 2 score: {score_array[1]}",
        f"dart 3 score: {score_array[2]}",
        f"total score: {total_score}"
    ]

    # Draw a solid rectangle behind the score tally for easier reading
    cv.rectangle(final_with_score, (final_with_score.shape[1] - 410, 0), (final_with_score.shape[1] - 10, 160), (0, 0, 0), -1)

    # Place text in the top right corner of the image
    y0, dy = 30, 30
    for i, line in enumerate(score_text):
        y = y0 + i * dy
        cv.putText(final_with_score, line, (final_with_score.shape[1] - 400, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #================================#

    #==========Export Files==========#

    #Save a copy of the images
    if file_export:
        cv.imwrite(os.path.join(output_dir, "input_image_with_dart_centers.png"), input_copy)
        cv.imwrite(os.path.join(output_dir, "extracted_board.png"), ref_masked)
        cv.imwrite(os.path.join(output_dir, "extracted_board_input.png"), input_masked)
        cv.imwrite(os.path.join(output_dir, "input_image_with_mask_applied.png"), input_masked)
        cv.imwrite(os.path.join(output_dir, "difference_image.png"), diff_thresh)
        cv.imwrite(os.path.join(output_dir, "highlighted_differences.png"), highlighted_diff)
        cv.imwrite(os.path.join(output_dir, "reference_Image.png"), reference)
        cv.imwrite(os.path.join(output_dir, "original_input_Image.png"), input_img)
        cv.imwrite(os.path.join(output_dir, "Final_Image.png"), input_with_zones)
        cv.imwrite(os.path.join(output_dir, "Final_Image_w_Score.png"), final_with_score)

    #================================#

    #==========Print Output==========#

    # Highlight the differences on the input image
    if verbose:
        cv.imshow("input image with dart centers", input_copy)
        cv.imshow("extracted board", ref_masked)
        cv.imshow("extracted board input", input_masked)
        cv.imshow("input image with mask applied", input_masked)
        cv.imshow("difference image", diff_thresh)
        cv.imshow("highlighted differences", highlighted_diff)
        cv.imshow("reference Image", reference)
        cv.imshow("original input Image", input_img)
        cv.imshow("Final Image", input_with_zones)
        cv.imshow("Final Image with Score", final_with_score)
    else:
        cv.imshow("Final Image with Score", final_with_score)

    print(f"Total Score: {total_score}")

    #================================#

    cv.waitKey(0)

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
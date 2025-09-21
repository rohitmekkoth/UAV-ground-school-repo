import cv2

# Read and display image to ensure the jpg file works
image = cv2.imread('apple.jpg')
cv2.imshow("Original Image", image)
cv2.waitKey(0)  # click a key to view the next displayed image

# Convert image to hues (hsv)
imageHue = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# create upper and lower bounds for the masks
# red for apple -> hue values are between 0-180 and red seems to curl around the end
lower1 = (0, 60, 60)
upper1 = (15, 255, 255)
lower2 = (170, 100, 100)
upper2 = (180, 255, 255)

# green for leaf
lower3 = (35, 75, 75)
upper3 = (85, 255, 255)

# brown for stem -> treated like a dark orange
lower4 = (10, 100, 20)
upper4 = (20, 255, 200)


# masks
mask1 = cv2.inRange(imageHue, lower1, upper1)
mask2 = cv2.inRange(imageHue, lower2, upper2)
redmask = mask1 | mask2

greenmask = cv2.inRange(imageHue, lower3, upper3)

brownmask = cv2.inRange(imageHue, lower4, upper4)

# A mask for the background is everything outside the above masks
# combine color masks into one object mask and invert it
objectmask = redmask | greenmask | brownmask
background_mask = cv2.bitwise_not(objectmask)

# overlay object mask onto image
result = cv2.bitwise_and(image, image, mask=objectmask)

cv2.imshow("Red Mask", redmask)
cv2.imshow("Green Mask", greenmask)
cv2.imshow("Brown Mask", brownmask)
cv2.imshow("Background mask", background_mask)
cv2.imshow("Result", result)
cv2.waitKey(0)

# Finding Center
# Find contours for each color
redcontours, _ = cv2.findContours(
    redmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
greencontours, _ = cv2.findContours(
    greenmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
browncontours, _ = cv2.findContours(
    brownmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# draw contours onto original image
cv2.drawContours(image, redcontours, -1, (0, 0, 255), 2)
cv2.drawContours(image, greencontours, -1, (0, 255, 0), 2)
cv2.drawContours(image, browncontours, -1, (42, 42, 165), 2)
# display contoured image
cv2.imshow("Contoured image", image)
cv2.waitKey(0)

# center for each contour
contours = [("Red", redcontours), ("Green", greencontours),
            ("Brown", browncontours)]
for color, contour in contours:
    biggest_area = 0
    for c in contour:
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            area = cv2.contourArea(c)
            if area > 500:  # only keep centroids that belong to a large contour
                # second layer of defense to prevent contours over threshold but less than the biggest
                biggest_area = max(area, biggest_area)
                if (area == biggest_area):
                    print(f" {color} Center: ({cx}, {cy})")
                    cv2.circle(image, (cx, cy), 5, (255, 255, 255), -1)
                    cv2.putText(image, f"({cx}, {cy})", (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1)
# display centers on contoured image
cv2.imshow("Centers", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

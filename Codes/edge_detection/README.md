##cSimple_Canny_method.py

* Using roi to remove the unnecessary parts of the image.

* Using hough transform to detect the line after Canny edge detection.

This method doesn't work well when it comes to cases where there are shadows, and can be easily influenced by noise.



## Fine_Sobel_method.py

* Using perspective transform to get a bird-viewed image
* Using Sobel to detection the edge in x-direction.

Still face the problem of noise.



## Color_based_method.py

* Using roi.
* Perspective transform.
* Using color selection to get the white and yellow regions.
* Using slide windows to detect left and right lane.

Sometimes, the start point of slide windows can change dramatically along the time.



##Color_based_method_for_video.py

* Add a search method that find the potential lane pixels around the found lane.
* Inverse perspective transform to make a result image where the lane is drawn.
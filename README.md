# Text-Detection-in-Netflix-Bitrate-menu-OCR-using-Tesseract-
In this project I manage to detect text in image with network performance values during Netfilx streaming.
I learned how to detect individual characters and words and how to place bounding boxes around them.
Using Python-tesseract an optical character recognition (OCR) tool for python and image processing techniques I succeeded to recognize and “read” the text in Netflix movies and export those values to CSV file.

*To open Netflix bitrate menu press Shift + Alt + Q on keyboard. Use screen recorder tool to record Netflix movie.

This tool performs the following steps for each frame:

1. Convert the image to GRAY format then convert it to binary image.
![image](https://user-images.githubusercontent.com/50642442/134968210-64c1c09f-7aaf-4556-9c9b-4a7abc933df8.png)

2.Find the borders with Canny function and cover it with two black lines (pre-process before dilate image).

![image](https://user-images.githubusercontent.com/50642442/134968419-038dd2fa-84cb-4f95-b51b-0135b00954f5.png)

3.Increase the white region in the image with dilate function. Kernel with smaller value like (2, 6) will detect
    each word instead of a line.

![image](https://user-images.githubusercontent.com/50642442/134968888-315e2fe1-5cc6-4902-a4f0-577c577d9108.png)

4.Find labels with connectedComponents function and plot rectangle around each line.

![image](https://user-images.githubusercontent.com/50642442/134969787-3bd65259-88a7-463b-9ee7-0ce009d5360d.png)

5. Send lines to Tesseract OCR model and convert it to text.
6. Create dict with all the network values using pandas library and save to CSV file. 

![image](https://user-images.githubusercontent.com/50642442/134971549-ca50b7c6-997e-45b6-9ecb-a5cdf6ffc6ce.png)





# Microscopic Images Analysis 👋
Hello everyone! I am Nerojeni, and this is my first professional project completed during my internship at the PIMM laboratory, which is affiliated with three institutions:

 - École Nationale Supérieure des Arts et Métiers
- CNRS
- Le CNAM

A french version is available in the repository, it's a file in PDF. 
      
## Introduction
I developed a Python application to analyze microscopic images. I programmed mathematical functions based on an article to study the quality of images.

[Read the Article](https://amnl.mie.utoronto.ca/data/J7.pdf) - Autofocusing Algorithm Selection in Computer Microscopy

## 🧐 Features
- Analyze Image Quality
- Find the optimal and non-optimal images from a list of images
- Identify the parameters used to create optimal images

## 🛠️ Install Dependencies    
```bash
pip install pybase64
pip install collection
pip install DateTime
pip install Dash
pip install dash-bootstrap-components
pip install Python-IO
pip install jsonlib
pip install matplotlib
pip install numpy
pip install os-sys
pip install pandas
pip install pathlib
pip install PyWavelets
pip install scipy
pip install scikit-image
pip install wordcloud
```

## 🛠️ Tech Stack
- [Dash](https://dash.plotly.com)
- [NumPy](https://numpy.org/doc/stable/)
- [Scikit-Image](https://scikit-image.org/)
- [Parallel Coordinates](https://plotly.com/python/parallel-coordinates-plot/)

## 🧑🏻‍💻 Usage

This application can be executed locally and accessed via the following web address: http://127.0.0.1:8050

1. Clone the repository
2. Install the dependencies
3. Run the application

## 💻Functionality of the different layouts      

### A. Home 

In this section, users are able to import images and associate parameters files to analyze them. 
<p> A button for download a Json file will apear if there is no Json file among the imported files</p>
<p>Json file regroups all the information about all the images. Indeed, we can find in the Json file, parameters of each images and the result of each functions. In this way, the importing process will be less longer, thanks to the json file.</p> 

### B. Dashboard

In this layout, the results of all functions can be viewed either simultaneously or individually. Indeed, there are two distinct graphs, where you can see the difference in results between the various functions or look at the results function by function. Additionally, it is also possible to get a preview of the image by clicking on the points on the curve.

### C. Image Parameters

This tab is dedicated to understanding the relationship between an optimal image and the parameters used to create it. To achieve this, I used the function results to display the parameters according to the results obtained.

### D. All Images

This layout as indicated in the title display all images imported by the user. 

## 😉 Explanation of the different functions 

<h3>A. Derivative-Based Algorithms</h3>

<h4>Generality </h4>

<p> For each functions, the final result is divided by the maximal result in a list of images. So they are noramlized and the results are between 0 and 1. </p>

<p> These functions are developed to study focus in optical microscope images. Therefore, it is important to consider this aspect to understand the results. The functions can be modified according to individual needs. </p>

<h4>1. F_1 : </h4>

<p>This first function calculates the sum of the absolute differences of consecutive lines divided by the maximum value of this sum. Thus, it allows observing interline variations for each image. </p>

<h4>2. F_2 : </h4>

<p> This function is similar to the previous one, but the sum of the differences is squared.</p>

<h4>3. F_3 : </h4>

<p>This function captures changes in pixel intensity variations. Therefore, it allows the detection of texture or patterns in the image. </p>

<h4>4. F_4 : </h4>

<p>This function uses a convolution filter called the Sobel operator to detect edges, meaning edge intensities. A high value would indicate a strong presence of edges; thus, an image with many details and sharp contours. Conversely, a low value would rather indicate a homogeneous or even blurry image. </p>

<h4>5. F_5 : </h4>

<p>This function calculates a measure of variation in a list of images using gradients and second derivatives of the images. Dx and Dy represent the horizontal and vertical variations of pixel intensities, respectively. Lxx and Lyy capture the curvature of the image in the x and y directions. </p>
<p>It returns the sum of the squares of the second derivatives as a measure of the intensity of curve and texture variations in the image divided by the maximum value of this sum. A high value indicates fine details in the image. </p>

<h4>6. F_6 : </h4>

<p>It detects texture variations using a Laplacian filter (a differential operator highlighting areas of rapid pixel intensity variations such as edges and textures). </p>
<p>It returns the sum of the convolution values divided by the maximum value of these convolutions. A high value indicates a strong texture variation in the image.</p>

<h4>7. F_7 : </h4>

<p> This function uses the wavelet transform, a mathematical technique to decompose a function or signal into components at different scales and resolutions.</p>

DWT decomposes the image into 4 sub-parts:

- LL: low-frequency component in the horizontal and vertical directions (approximate);
- LH: low-frequency component in the horizontal direction and high-frequency component in the vertical direction (horizontal details);
- HL: high-frequency component in the horizontal direction and low-frequency component in the vertical direction (vertical details);
- HH: high-frequency component in both directions (diagonal details).

<p>It returns the sum of the absolute values of these detail coefficients (LH, HL, HH) {measuring the intensity of texture and detail variations in the image} divided by the maximum value among the absolute values of the coefficients. A high value indicates a strong presence of details and textures in the image and vice versa. </p>

<h4>8. F_8 : </h4>

<p> A function that uses wavelet decomposition with mean adjustment. It is similar to F_7 but the sum of the coefficients is different. Here, the difference between the absolute value of the coefficient and the mean of the absolute value of the coefficient is calculated.</p>

<p>It returns the sum of the squared differences divided by the product of the image size and divided by the maximum value among this list of sums.</p>

<h4>9. F_9 : </h4>

<p>Identical to the previous one but without the absolute values. </p>

<h3>B. Statistics-Based Algorithms </h3>

<h4>10. F_10 : </h4>

<p>It calculates the variance of the pixels of each image. More precisely, the dispersion of pixel intensities around the mean by quantifying the variability of the image's intensity.</p>

<p>A high value indicates a strong variability of pixel intensity == marked contrasts / significant variations. </p>

<h4>11. F_11 : </h4>

<p>Similar to F_10 but calculates the relative variance. A high value means there is a strong relative variability of pixel intensity compared to the mean, indicating marked contrasts. </p>

<h4>12. F_12 : </h4>

<p>This function calculates the difference between the original image and two shifted versions of the image. This helps to understand the change in pixel intensity relative to its immediate neighbor and the one located two pixels away. </p>

<p>It first calculates the sum of the products of the original image and the image shifted by 1 and then by 2. Then it calculates the difference between these sums, not forgetting to normalize the result.</p>

<h4>13. F_13 : </h4>

<p>It studies a correlation based on the standard deviation of adjacent pixels. It first calculates the sum of the products of adjacent pixels along the chosen axis. Then, it subtracts from this sum the product of the image size and the squared mean of this image.</p>

<h3>C. Histogram-Based Algorithms </h3>

<h4>14. F_14 : </h4>

<p>This function simply calculates the difference between the maximum and minimum intensities of an image. </p>

<h4>15. F_15 : </h4>

<p>This function calculates the entropy of an image by measuring the amount of information contained in the image through the histogram of pixel intensities. </p>

<h4>16. F_16 : </h4>

<p>This function aims to calculate the number of pixels whose intensity is greater than or equal to the image mean. </p>

<h4>17. F_17 : </h4>

<p>It counts the number of pixels whose intensity is less than or equal to the image mean</p>

<h4>18. F_18 : </h4>

<p>It calculates the sum of the squares of pixel intensities above the mean. </p>

## 🌟 Future Work 

- Improve functions to adapt them to every type of images, not just microscopic ones
- Develop more visualizations

## 🙇 Author

[LinkedIn](https://www.linkedin.com/in/nerojeni-sivarajah-14656a265/)







## Objectives

* Apply Template Matching method using different similarity metrics.
* Apply SIFT and matching images with different rotations and scales.
* Reporting results and computation times.


### A) Computer Vision Functions

 need to implement Python functions which will support the following tasks:

1. Match the image set features using:
    1. Correlation
    2. Zero-mean correlation
    3. Sum of squared differences (SSD)
    4. and normalized cross correlations.**Then report matching computation time in the GUI.**


2. Generate feature descriptors using scale invariant features (SIFT).**Report computation time in the GUI.**


You should implement these tasks **without depending on OpenCV library or alike**.


organize  implementation of the core functionalities:

1. `CV404Template.py`: this will include the implementation for template matching functions (requirement 1). You can use the distance functions from the section as they are. Develop your own methods to extract the similar objects.
2. `CV404SIFT.py`: this will include the implementation for SIFT technique (requirement 2). Gather the pieces of codes in the notebook in an organized Python class.

### B) GUI Integration

Integrate your functions in part (A) to the following Qt MainWindow design:



| Tab 7 |
|---|
| <img src=".screen/tab7.png" style="width:500px;"> |

| Tab 8 |
|---|
| <img src=".screen/tab8.png" style="width:500px;"> |

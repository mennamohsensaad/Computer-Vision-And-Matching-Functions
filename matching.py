import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
from scipy import misc

# Functions
def matchTemplate_Correlate(image, template):
	y = np.empty(image.shape)
	y = correlate2d(image, template, 'same')
	return y

def matchTemplate_Correlate_ZMean(image, template):	
	return matchTemplate_Correlate(image, template - template.mean())

# Direct 2D correlation of the image with template
input_img = misc.face(gray=True) - misc.face(gray=True).mean()
input_temp = np.copy(input_img[300:365, 670:750])  # right eye

# 2D correlation
result = matchTemplate_Correlate(input_img, input_temp)

# 2D correlation with zero mean
resultZ = matchTemplate_Correlate(input_img, input_temp)

# plotting the result
y, x = np.unravel_index(np.argmax(result), result.shape)  # find the match


fig, (ax_orig, ax_template, ax_corr, ax_corrZ) = plt.subplots(4, 1, figsize=(6, 15))
ax_orig.imshow(input_img, cmap='gray')
ax_orig.set_title('Original')
ax_orig.set_axis_off()
ax_template.imshow(input_temp, cmap='gray')
ax_template.set_title('Template')
ax_template.set_axis_off()
ax_corr.imshow(result, cmap='gray')
ax_corr.set_title('Cross-correlation')
ax_corr.set_axis_off()
ax_corrZ.imshow(resultZ, cmap='gray')
ax_corrZ.set_title('Cross-correlation(Zero Mean)')
ax_corrZ.set_axis_off()
ax_orig.plot(x, y, 'ro')
fig.show()



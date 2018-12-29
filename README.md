# OpticalAI


# GazeCapture Dataset 

The problem was to create a neural network which given an image of a user looking at their phone (from the front-facing camera) would predict where a user was looking on their phone.

This dataset consisted of images taken from the front-facing camera of a phone while users were looking at a circle on their phone, and the coordinates of the circle. 

To ensure that the results could be applied to any device, the coordinates were predicted with respect to the camera and not the pixel coordinates. The error is measured as the mean absolute error between the predicted coordinates and the true coordinates.

The previous best recorded average error was 1.66cm (Kannan 2017). My convolutional neural network (CNN) architecture achieves an average error of 1.38cm on the portrait dataset.

# Implementation
The architecture consists of three MobileNetV2 models, where each model was only trained on a single orientation (portrait, landscape home button on the left, and landscape home button on the right).

All models were trained for ~8 epochs with the following specifications,
- Loss: logcosh
- Optimizer: RMSProp
- Initial Learning Rate: 0.002 

# Intuition

## Model Per Orientation
Training separate models for each orientation would lead to less data manipulation (and possible loss of data for the model), and would not increase computation time since the phone orientation could be found out at runtime. 

## Image Size
Previous research found that as the image size increased, the centimetre error would decrease (Kannan 2017). Since we are using a smaller model (MobileNet) than used previously (Kannan 2017) in general we can increase the image size without a significant loss of runtime efficiency. The models were trained on a size of (667, 375, 3) for portrait and (375, 667, 3) for landscape. These sizes corresponded to the largest and most frequent in the dataset.

## Loss Function
The loss function chosen was Logcosh, compared to the previous research which utilized the L1 loss function (Kannan 2017). Logcosh is approximately equal to `x^2 / 2` when x is small, and `|x|` when x is large. Since the dataset was crowdfunded I expected there would be outliers in the dataset (people not looking where they should be). This loss function was chosen since it provides the same loss for outliers (e.g. model correctly predicts a user looking away from the screen but data says they are looking at the screen) but will more heavily punish small model prediction errors. Hence it was expected that higher accuracy could be achieved.

# Results 

After training the portrait model to an accuracy of ~1.4cm test error, I looked at which images were being predicted incorrectly (> 7cm error). The majority of these cases were users looking away from their phones or users wearing glasses which caused a glare covering their eyes. By removing 37 outlier cases of the 1,290 total cases, the results improved to a  ~1.38cm test error. 

Training the portrait model on the same data split used in previous research (Krafka et.al. 2016) the final test error was 1.38cm. The landscape model was trained on a different data split from the paper and also achieved a final test error of 1.38cm.

# References
Kannan, H., Eye tracking for the iPhone using deep learning, MIT MASc Thesis, https://dspace.mit.edu/handle/1721.1/113142, 2017

K.Krafka, A. Khosla, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba, Eye Tracking for Everyone, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

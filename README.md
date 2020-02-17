# Recognition of hand written digits
<h3>Table of Content:</h3>
</br>
<ul>
  <li>Introduction</li>
  <li>Logistic Regression</li>
  <li>Neural Network Classifier</li>
  <li>Results</li>
  <li>Data Source</li>
 </ul>
<p>
<h3>1) Introduction</h3></br> Samples provided from Kaggle dataset includes handwritten digits total of 70,000 images consisting of 42,000 examples in training set and 28,000 examples in testing set, with labeled training images from 10 digits (0 to 9). Created different classifiers to compare the results between each other and also by parameter tuning.
 </br>     
<h3>2) Logistic Regression</h3></br> Initially baselined the results using a logistic regression multi-class model. Analysed the effect of cost parameter and chose the best parameters to get maximum accuracy on test dataset. </br>

<h3>3) Neural Net Classifier</h3></br>Built a CNN model using multiple hidden layers and finalised the 2 layer neuron as it has given better results compared to higher number of layers. The model includes pooling layers, drop outs as well. 
   </br>
<h3>4) Results</h3></br>   Logistic regression model resulted in a 91% accuracy in predicting the hand drawn digit images. The test accuracy has increased from 91% to 97% when CNN model was implemented. </br></br>
<h3>5) Data Source</h3></br>The dataset has been taken from a kaggle competition. The below link can be used to get the test and train datasets which were used in this project.
 https://www.kaggle.com/c/digit-recognizer/data
 </p>

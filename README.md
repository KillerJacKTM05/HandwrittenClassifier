# HandwrittenClassifier
Handwritten Digit Classifier AI work directory. Keras library is planned to use.

# Version 0.1
GPU-CPU monitoring, epoch base training code added. A simple sequential keras network added for model training. F1 and accuracy metrics added.
Not yet tested.

# Version 0.2
Keras-torch splitted files added. Because, i couldn't solve dependency problem for keras dependent libraries.
Not yet tested, (laptop issues)

# Version 0.3
Small changes on keras.py file. It's executable and it looks like it started to train successfully in pre-tests.
When epoch 250 training is made with using DIDA 70k dataset, results will be added.
More information for dataset: https://didadataset.github.io/DIDA/

# Version 1.0
Torch version deprecated. Old keras model was unable to learn fast & eligible. Remodeled keras sequential. Random image display for each training iteration is added to visualize whether the training data is appropiate or not.
Even the 5 epoch training has 95.42% weighted F1 score, avg accuracy of 95.43%.
Worst class is 8, best class is 7 even if it's recall and precision has some gap between.
In Summary:
The model is performing very well. The metrics suggest a high level of confidence in the model's predictions, with minimal class bias.
Potential Next Steps:
- Error Analysis: Delve deeper into misclassifications. For instance, which numbers are commonly misclassified as 8? This can provide insights into potential areas for improvement.

- Data Augmentation & Regularization

- Model Architecture: Depending on the problem's complexity, we might benefit from a more sophisticated model architecture, such as a convolutional neural network (CNN).

- External Testing: Using an external test set (if available) to ensure its generalization capability.

- Model Interpretability: Tools like SHAP, LIME, or Grad-CAM can help us understand which parts of your input images contribute most to the model's decisions.
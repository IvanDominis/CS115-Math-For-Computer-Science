1. Merge 2 datasets: cleveland and reprocessed.hugarian.data

2. Split 80% for training and remaining for testing

3. Normalization on training data

4. Normalization on testing data followed training data

5. Train logistic regression for 2 classes: 0 and 1

--> using binary cross entropy loss

6. Evaluate on: Confusion matrix, precision, recall, f1, accuracy

7. After evaluation, if there is imbalanced problem

--> use weighted class cross entropy loss

8. Summerization

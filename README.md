## Introduction
This project aims to build an LSTM model to detect if an email or any other type of message is spam.<br>

## App
1. Run "python spam_detection.py" in terminal. <br>
2. Enter/paste messages into the text box and click Submit. <br>
3. A new page will display the typed message and detection result. <br>
4. The html files in the "template" folder is necessarily for the app to run. <br>

## Result
See the screenshots in the directories "app_result_screenshot/normal" and "app_result_screenshot/spam" for 3 examples of normal
emails/messages and 3 examples of spam emails/messages. <br>

## Model Training
Model is trained with the data set emails_spam_normal.csv which contains 5000 emails and T4-GPU in Google Colab. <br>
A downsizing sampling was performed before model training. <br>
spam_detection.json and spam_detection.weights.h5 together contain the full info of the trained model. <br>
These two files are loaded in spam_detection.py for the app to make detections of an unseen email/message. <br>

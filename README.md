# BankCardRecognizer
This is our Machine Learning Final Project. Our task is to recognize Bank Card Number using Android Application.

### Final Report
[Click here to download our final report](https://github.com/ardiya/BankCardRecognizer/raw/master/ML_report_Group20.pdf)

### Simple Explanation
Our solution consist of several parts, which are:
- Android Application to take image, upload to web server and show the result
- Web application that will run image segmentation and recognize digits
- Compiled C++ application that do the image segmentation


![flowchart.png](https://github.com/ardiya/BankCardRecognizer/raw/master/images/flowchart.png)


**Our final android application looks like this.**

![flowchart.png](https://github.com/ardiya/BankCardRecognizer/raw/master/images/android.PNG)


**Our web application can also show some result but with JSON output** (Yes, it is used for communicating with the Android app).

![result.png](https://github.com/ardiya/BankCardRecognizer/raw/master/images/web.PNG)
![webresult.png](https://github.com/ardiya/BankCardRecognizer/raw/master/images/webresult.png)
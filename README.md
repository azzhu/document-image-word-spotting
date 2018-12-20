
## INPUT and OUTPUT
input：a image（For example, a handwritten picture of an English composition）

output：many image lists（Represents each line of the English composition in the input image）

The effect is as follows：
![input](pic/Snipaste_2018-10-20_22-52-08.jpg "input")

## Brief Description

This project is mainly aimed at the preprocessing part of English handwriting recognition image. When doing handwriting recognition, it will be very easy if the images have been cut as above, but often the most difficult part is not to design the neural network of handwriting recognition, but the image preprocessing part in front of it, how to cut out the images in various cases by lines. That's what this project is all about.

Of course, not only English, handwritten Chinese composition can also be cut out; If it's printed instead of handwritten, it's easier to cut.

## Algorithm Principle
Heatmap is mainly divided into two steps. The first step is to calculate a heatmap (text area heatmap). In the second step, heatmap is used to plan the rows.

Core idea: the accumulation of rules.

How the specific principle, look at the code, very simple, anyway, is a variety of rules.

Heatmap was calculated in the first step. The unet network was used for deep learning, and the effect was quite good. However, the speed was a little slower.

This code is written primarily in Python, mainly to meet the needs of other projects. Of course, if you use C++, it will definitely be faster. In fact, I wrote it in C++ (including C++ code) at first, and later converted it into Python code. This code was written very early, I just learned Python at that time, so the code is very messy and has a lot of c++ style (manual face cover), but please ignore, just look at the effect.

* ![img](pic/Selection_011.bmp "img")
* ![img](pic/Selection_012.bmp "img")
* ![img](pic/Selection_009.bmp "img")
* ![img](pic/Selection_010.bmp "img")
* ![img](pic/Selection_013.bmp "img")
* ![img](pic/Selection_014.bmp "img")

## More Experimental Results

* ![img](pic/Snipaste_2018-10-20_22-30-10.jpg "img")
* ![img](pic/Snipaste_2018-10-20_22-29-06.jpg "img")

* ![img](pic/Snipaste_2018-10-20_22-34-39.jpg "img")
* ![img](pic/Snipaste_2018-10-20_22-34-05.jpg "img")

## Please leave a message if you have any questions
If you have any questions in use, please feel free to give me feedback. You can also communicate with me through the following contact information

* E-mail: 228812066@qq.com
* QQ: 228812066



#include <conio.h>

#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;


int main()
{
	
	Mat frame;

	VideoCapture cap;
	cap.open(0);

	cap.read(frame);


	imshow("test", frame);

	


	frame.convertTo(frame, CV_32S);
	cout << frame.at<int>(100, 100);
	waitKey();

	return 0;


}
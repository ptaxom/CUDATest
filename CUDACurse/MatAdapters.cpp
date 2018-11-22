//#include "Kernel.h"
//
//using namespace cv;
//
//int *MatToArray(cv::Mat &mat)
//{
//	int *arr = new int[mat.cols * mat.rows];
//	for(int i = 0; i < mat.rows; i++)
//		for (int j = 0; j < mat.cols; j++)
//		{
//			int r = mat.at<cv::Vec3b>(i, j).val[0] << 16;
//			int g = mat.at<cv::Vec3b>(i, j).val[1] << 8;
//			int b = mat.at<cv::Vec3b>(i, j).val[2];
//			arr[i * mat.cols + j] = r + g + b;
//		}
//	return arr;
//}
//
//
//cv::Mat &ArrayToMat(int *arr, int cols, int rows)
//{
//	cv::Mat img(rows, cols, CV_8U);
//	for(int i = 0; i < rows; i++)
//		for (int j = 0; j < cols; j++)
//		{
//			int pixel = arr[i * cols + j];
//			int r = (pixel >> 16) & 0x000000ff;
//			int g = (pixel >> 8) & 0x000000ff;
//			int b = pixel & 0x000000ff;
//			cv::Vec3b vec(r, g, b);
//			img.at<cv::Vec3b>(i, j) = vec;
//		}
//	delete arr;
//	return img;
//}
//
//
//cv::Mat &BoxBlur( cv::Mat &src)
//{
//	float *kern = new float[9];
//	for (int i = 0; i < 9; i++)
//		kern[i] = (float)1.0f / 9.0f;
//	int *arr = MatToArray(src);
//	int kernelHalf = 1;
//	int cols = src.cols;
//	int rows = src.rows;
//	int *out = nullptr;
//	kernelGPU(arr, kern, kernelHalf, cols, rows, out);
//	return ArrayToMat(out, cols, rows);
//}
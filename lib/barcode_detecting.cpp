#include<opencv2/opencv.hpp>
#include<algorithm>
#include <cstdlib>
#include<math.h>
#include<utility>

void print(cv::Mat src, std::string window_name)
{
	cv::namedWindow(window_name, cv::WINDOW_NORMAL);
	cv::resizeWindow(window_name, 600, 400);
	imshow(window_name, src);
}
cv::Mat filtred(cv::Mat src_bin)
{
	cv::Mat src_filtred;
	cv::Point anchor = cv::Point(-1, -1);
	double delta = 0;
	int ddepth = -1;
	cv::Mat kernel;
	kernel = cv::Mat::ones(3, 3, CV_32F) / (float)(3 * 3);
	filter2D(src_bin, src_filtred, ddepth, kernel, anchor, delta, cv::BORDER_DEFAULT);

	return src_filtred;
}

cv::Mat sharpering(cv::Mat src_filtred)
{
	cv::Mat src_sh;
	cv::Point anchor = cv::Point(-1, -1);
	double delta = 0;
	int ddepth = -1;
	cv::Mat kernel = (cv::Mat_<double>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1); 
	filter2D(src_filtred, src_sh, ddepth, kernel, anchor, delta, cv::BORDER_DEFAULT);
	return src_sh;
}


std::vector<std::vector<cv::Point>>  sortContours(std::vector<std::vector<cv::Point>> & contours)
{
	for (size_t i = 0; i < contours.size() - 1; i++) {
		for (size_t ind = 0; ind < contours.size() - 1; ind++)
		{
			if (cv::contourArea(contours[ind]) > cv::contourArea(contours[ind+1]))
			{
				std::vector<cv::Point> tmp;
				tmp = contours[ind];
				contours[ind] = contours[ind + 1];
				contours[ind + 1] = tmp;
			}
		}
	}
	return contours;
}

double lineLenght(cv::Point2f a, cv::Point2f b)
{
	return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

int main()
{
	cv::Point anchor = cv::Point(-1, -1);
	double delta = 0;
	int ddepth = -1;
	cv::Mat kernel;
	kernel = cv::Mat::ones(3, 3, CV_32F) / (float)(3 * 3);
	cv::Mat img = cv::imread("E:/Projects/barcode_detecting/data/test6.jpg", cv::IMREAD_COLOR);
	cv::Mat src_gray, src_bin, src_fil, src_sh;
	cv::cvtColor(img, src_gray, cv::COLOR_BGR2GRAY);
	src_fil = filtred(src_gray);
	src_sh = sharpering(src_gray);
	cv::threshold(src_sh, src_bin, 128, 255, cv::THRESH_OTSU);
	cv::RNG rng(12345);
	print(img, "barcode");
	//print(src_gray, "barcode_gray");
	//print(src_bin, "barcode_bin");
	//print(src_fil, "barcode_fil");
	//print(src_sh, "barcode_sh");
	
	//src_gray = src_bin;
	cv::Mat grad_x, grad_y, grad;
	cv::Mat abs_grad_x, abs_grad_y;
	int scale =1;
	//ddepth = CV_32F;
	Sobel(src_gray, grad_x, ddepth, 1, 0, -1, scale, delta=0, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);


	Sobel(src_gray, grad_y, ddepth, 0, 1, -1, scale, delta=0, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	cv::subtract(grad_x, grad_y, grad);
	//grad = 0.5 * grad_x + 0.5 * grad_y;
	cv::convertScaleAbs(grad, grad);
	cv::threshold(grad, grad, 200, 255, cv::THRESH_OTSU);
	print(grad_y, "grad_y");
	print( grad_x,"grad_x");
	print(grad, "grad");
	cv::Mat blured, blured_bin;
	cv::GaussianBlur(grad, blured, cv::Size(9, 9), 0, 0, cv::BORDER_DEFAULT);
	//cv::blur(grad, blured, cv::Size_<int>(9, 9));
	//blured = grad;
	print(blured, "blur");
	
	cv::threshold(blured, blured_bin, 200, 255, cv::THRESH_OTSU);
	int size_x = (img.cols / 600) * 3;
	int size_y= (img.rows / 450) * 4;
	cv::erode(blured_bin, blured_bin, cv::Mat::ones(1, 1, CV_32F), cv::Point(-1, -1), 1);
	cv::dilate(blured_bin, blured_bin, cv::Mat::ones(1, 1, CV_32F), cv::Point(-1, -1), 2);
	cv::erode(blured_bin, blured_bin, cv::Mat::ones(1, 1, CV_32F), cv::Point(-1, -1), 1);
	cv::dilate(blured_bin, blured_bin, cv::Mat::ones(1, size_x, CV_32F), cv::Point(-1, -1), 3);
	cv::erode(blured_bin, blured_bin, cv::Mat::ones(1, size_x, CV_32F),cv::Point(-1,-1),3);
	cv::dilate(blured_bin, blured_bin, cv::Mat::ones(size_x/3, size_x/3, CV_32F), cv::Point(-1, -1), 2);
	cv::erode(blured_bin, blured_bin, cv::Mat::ones(size_x/3, size_x/3, CV_32F), cv::Point(-1, -1), 2);
	
	print(blured_bin, "erodil");

	kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size_<int>(img.cols/30, img.rows/30));
	//cv::morphologyEx(blured_bin, blured_bin, cv::MORPH_CLOSE, kernel);
	print(blured_bin, "blured_bin");
	//cv::Canny(blured_bin, blured_bin, 50, 200);
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hireachy;
	cv::findContours(blured_bin, contours, hireachy,cv::RETR_CCOMP,cv::CHAIN_APPROX_TC89_KCOS, cv::Point());
	cv::Mat drawing = cv::Mat::zeros(blured_bin.size(), CV_8UC3);
	int ncomp = contours.size();
	
	contours=sortContours(contours);
	
	/*for (int i = 0; i < ncomp; i++)
	{
		cv::drawContours(img, contours, (int)i, cv::Scalar(rng(255), rng(255), rng(255)),3);
	}*/
	print(blured_bin, "blured_bin");
	for (auto cont : contours)
	{
		std::cout << cv::contourArea(cont) << std::endl;
	}

	cv::RotatedRect minRect;

	cv::Scalar color = cv::Scalar(0,255,0);
		
	cv::Point2f rect_points[4];
	int imgArea = img.cols * img.rows;
	int cond = 0;
	int ind = ncomp-1;
	while (cond<4)
	{cond = 0;
		std::cout << " size:" << cv::contourArea(contours[ind]);
		minRect = minAreaRect(contours[ind]);
		minRect.points(rect_points);
		double area = cv::contourArea(contours[ind]) / (lineLenght(rect_points[0], rect_points[1]) * lineLenght(rect_points[1], rect_points[2]));
		for (size_t i = 0; i < 4; i++)
		{	
			
			if (rect_points[i].x < (double)blured_bin.cols && rect_points[i].y < (double)blured_bin.rows && rect_points[i].x >=0. && rect_points[i].y >= 0.)
				if (area > 0.7)
				{
					++cond;
					std::cout << "area: " << area << std::endl;
				}
			std::cout << " " << rect_points[i].x << " " << rect_points[i].y <<" cond:"<<cond<< std::endl;
			if (rect_points[i].y < 0)
				std::cout << "wtf" << std::endl;
		}
		--ind;
		if (ind == 0)
			cond = 4;
		std::cout << "col"<< blured_bin.cols << " " << blured_bin.rows << std::endl;
	}
		 
	/*minRect = minAreaRect(contours[ncomp]);
	minRect.points(rect_points);*/
		for (int j = 0; j < 4; j++)
		{
			line(img, rect_points[j], rect_points[(j + 1) % 4], color,3);
			//std::cout << rect_points[j];
		}

	print(img, "contours");
	
	cv::waitKey(0);
	return 0;
}
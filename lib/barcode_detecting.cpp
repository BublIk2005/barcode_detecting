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
//600x450
cv::Mat scalePyr(cv::Mat &src)
{
	cv::Mat res;
	int scale;

	if ((src.cols / 600)>=2) {
		scale = src.cols / 600;
		cv::pyrDown(src, res, cv::Size(int(src.cols / scale), int(src.rows / scale)), cv::BORDER_DEFAULT);
	}
	else if ((600 / src.cols) >= 2) {
		scale = 600 / src.cols;
		cv::pyrUp(src, res, cv::Size(int(src.cols * scale), int(src.rows * scale)), cv::BORDER_DEFAULT);
	}
	else
		res = src;
	std::cout << res.cols << " " << res.rows << std::endl;
	return res;
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

cv::Mat Rotation(cv::Mat img, cv::Point2f a, cv::Point2f b)
{
	cv::Point2f vec = a - b;
	cv::Point2f dir(img.cols, 0);
	float cos = (vec.x * dir.x + vec.y * dir.y) / (sqrt(vec.x * vec.x + vec.y * vec.y) * sqrt(dir.x * dir.x + dir.y * dir.y));
	float angle = 57.3*acosf(cos)-3;
	std::cout <<"cos: "<<cos<<" angle: "<< angle << std::endl;
	cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
	cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
	cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), img.size(), angle).boundingRect2f();
	rot.at<double>(0, 2) += bbox.width / 2.0 - img.cols / 2.0;
	rot.at<double>(1, 2) += bbox.height / 2.0 - img.rows / 2.0;

	cv::Mat result;
	cv::warpAffine(img, result, rot, bbox.size());
	return result;
}

std::vector<cv::Mat> Gradients(cv::Mat src_gray)
{
	std::vector<cv::Mat> gradients;
	cv::Mat grad_x, grad_y, gradX,gradY,grad45,grad135;
	cv::Mat abs_grad_x, abs_grad_y;
	int scale = 1;
	int ddepth = CV_32F;
	int delta;
	Sobel(src_gray, grad_x, ddepth, 1, 0, -1, scale, delta = 0, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	Sobel(src_gray, grad_y, ddepth, 0, 1, -1, scale, delta = 0, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	cv::subtract(grad_x, grad_y, gradX);
	cv::convertScaleAbs(gradX, gradX);
	gradients.push_back(gradX);
	cv::subtract(grad_y, grad_x, gradY);
	cv::convertScaleAbs(gradY, gradY);
	gradients.push_back(gradY);
	grad45 = 0.5 * grad_x + 0.5 * grad_y;
	cv::convertScaleAbs(grad45, grad45);
	gradients.push_back(grad45);
	grad135 = -0.5 * grad_x + 0.5 * grad_y;
	cv::convertScaleAbs(grad135, grad135);
	gradients.push_back(grad135);
	return gradients;
}
std::vector<cv::Mat> contours(cv::Mat & grad, int scale, char* UporDown) {
	cv::Mat blured, blured_bin;
	cv::GaussianBlur(grad, blured, cv::Size(9, 9), 0, 0, cv::BORDER_DEFAULT);

	cv::threshold(blured, blured_bin, 200, 255, cv::THRESH_OTSU);

	cv::erode(blured_bin, blured_bin, cv::Mat::ones(1, 1, CV_32F), cv::Point(-1, -1), 1);
	cv::dilate(blured_bin, blured_bin, cv::Mat::ones(1, 1, CV_32F), cv::Point(-1, -1), 2);
	cv::erode(blured_bin, blured_bin, cv::Mat::ones(1, 1, CV_32F), cv::Point(-1, -1), 1);
	std::array<int, 5> id{ 300,150,100,50,20 };
	std::vector<cv::Mat> result;
	for (int i = 0; i < 5; i++) {
		cv::Mat res;
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size_<int>(grad.cols / id[i], 1));
		cv::morphologyEx(blured_bin, res, cv::MORPH_CLOSE, kernel);
		result.push_back(res);
	}
	
	return result;
}
cv::Mat findBarcode(cv::Mat src)
{
	std::vector<std::pair<cv::Mat, cv::RotatedRect>> result;
	cv::Mat src_gray, src_bin, src_fil, src_sh;
	cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
	cv::RNG rng(12345);
	src_gray = scalePyr(src_gray);
	std::vector<cv::Mat> gradients = Gradients(src_gray);
	int scale=1;
	char* UporDown=" ";
	if ((src.cols / 600) >= 2) {
		scale = src.cols / 600;
		UporDown = "Up";
	}
	else if ((600 / src.cols) >= 2) {
		scale = 600 / src.cols;
		UporDown = "Down";
	}
	else scale = 1;

	for (int i = 0; i < gradients.size(); ++i) {
		contours(gradients[i], scale, UporDown);
	}


	delete UporDown;
	
	return src;
}



int main()
{
	cv::Point anchor = cv::Point(-1, -1);
	double delta = 0;
	int ddepth = -1;
	cv::Mat kernel;
	kernel = cv::Mat::ones(3, 3, CV_32F) / (float)(3 * 3);

	cv::Mat img = cv::imread("E:/Projects/barcode_detecting/data/test1.jpg", cv::IMREAD_COLOR);
	cv::Mat src_gray, src_bin, src_fil, src_sh;
	cv::cvtColor(img, src_gray, cv::COLOR_BGR2GRAY);
	src_fil = filtred(src_gray);
	src_sh = sharpering(src_gray);
	cv::threshold(src_sh, src_bin, 128, 255, cv::THRESH_OTSU);
	cv::RNG rng(12345);
	print(img, "barcode");

	src_gray=scalePyr(src_gray);

	
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
	//grad = grad_x*0.5 +  grad_y*0.5;
	cv::convertScaleAbs(grad, grad);
	cv::threshold(grad, grad, 200, 255, cv::THRESH_OTSU);
	print(grad_y, "grad_y");
	print( grad_x,"grad_x");
	print(grad, "grad");
	cv::Mat blured, blured_bin;
	cv::GaussianBlur(grad, blured, cv::Size(9, 9), 0, 0, cv::BORDER_DEFAULT);
	//cv::blur(grad, blured, cv::Size_<int>(9, 9));
	//blured = grad;
	//print(blured, "blur");
	
	cv::threshold(blured, blured_bin, 200, 255, cv::THRESH_OTSU);
	int size_x = (img.cols / 600) * 3;
	int size_y= (img.rows / 450) * 4;
	cv::erode(blured_bin, blured_bin, cv::Mat::ones(1, 1, CV_32F), cv::Point(-1, -1), 1);
	cv::dilate(blured_bin, blured_bin, cv::Mat::ones(1, 1, CV_32F), cv::Point(-1, -1), 2);
	cv::erode(blured_bin, blured_bin, cv::Mat::ones(1, 1, CV_32F), cv::Point(-1, -1), 1);
	/*cv::dilate(blured_bin, blured_bin, cv::Mat::ones(1, size_x, CV_32F), cv::Point(-1, -1), 3);
	cv::erode(blured_bin, blured_bin, cv::Mat::ones(1, size_x, CV_32F),cv::Point(-1,-1),3);
	cv::dilate(blured_bin, blured_bin, cv::Mat::ones(size_x/3, size_x/3, CV_32F), cv::Point(-1, -1), 2);
	cv::erode(blured_bin, blured_bin, cv::Mat::ones(size_x/3, size_x/3, CV_32F), cv::Point(-1, -1), 2);*/
	
	//print(blured_bin, "erodil");

	kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size_<int>(img.cols/100, 1));
	cv::morphologyEx(blured_bin, blured_bin, cv::MORPH_CLOSE, kernel);
	//print(blured_bin, "blured_bin");
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
	
	//print(blured_bin, "blured_bin");
	/*for (auto cont : contours)
	{
		std::cout << cv::contourArea(cont) << std::endl;
	}*/

	cv::RotatedRect minRect;

	cv::Scalar color = cv::Scalar(0,255,0);
		
	cv::Point2f rect_points[4];
	int imgArea = img.cols * img.rows;
	int cond = 0;
	int ind = ncomp-1;
	while (cond<4)
	{cond = 0;
		//td::cout << " size:" << cv::contourArea(contours[ind]);
		minRect = minAreaRect(contours[ind]);
		minRect.points(rect_points);
		double area = cv::contourArea(contours[ind]) / (lineLenght(rect_points[0], rect_points[1]) * lineLenght(rect_points[1], rect_points[2]));
		for (size_t i = 0; i < 4; i++)
		{	
			
			if (rect_points[i].x < (double)blured_bin.cols && rect_points[i].y < (double)blured_bin.rows && rect_points[i].x >=0. && rect_points[i].y >= 0.)
				if (area > 0.6)
				{
					++cond;
				}
		}
		--ind;
		if (ind == 0)
			cond = 4;
	}
	if ((img.cols / 600) >= 2) {
		scale = img.cols / 600;
		for (size_t i = 0; i < 4; i++)
		{
			rect_points[i] = rect_points[i] * scale;
		}
	}
	else if ((600 / img.cols) >= 2) {
		scale = 600/img.cols;
		for (size_t i = 0; i < 4; i++)
		{
			rect_points[i] = rect_points[i] / scale;
		}
	}
	cv::Mat contourIm;
	//contourIm = img;
	cv::Rect cont(rect_points[0], rect_points[1]);
	cv::copyTo(img, contourIm, img);
		for (int j = 0; j < 4; j++)
		{
			line(contourIm, rect_points[j], rect_points[(j + 1) % 4], color,3);
		}

		

	print(contourIm, "contours");
	cv::Mat rot, rotImg;
	cv::copyTo(blured_bin, rot, blured_bin);
	//rot = Rotation(contourIm, rect_points[2], rect_points[3]);
	//print(rot, "Rotation");
	//cv::Rect2f bbox = minRect;
	//cv::Rect roi(rect_points[2], rect_points[3]);
	cv::Rect roi = minRect.boundingRect2f();
	cv::Mat imgRoi = rot(roi);
	rotImg = img(roi);
	print(imgRoi, "ROI1");
	

	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(imgRoi.cols/10, imgRoi.cols/10));
	cv::morphologyEx(imgRoi, imgRoi, cv::MORPH_CLOSE, element);
	print(imgRoi, "ROI");
	
	cv::Mat dist;
	cv::distanceTransform(imgRoi, dist, cv::DIST_L2, 3, CV_32F);

	 
	cv::normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);
	//std::cout << imgRoi.type() << " " << dist.type()<<" " << CV_8U << std::endl;
	print(dist, "distT");
	
	cv::threshold(dist, dist, 0.99, 1.0, cv::THRESH_BINARY);
	cv::normalize(dist, dist, 255, 0, cv::NORM_MINMAX);
	print(dist, "distTr");
	cv::Mat dist8u;
	dist.convertTo(dist8u, CV_8U);
	//cv::cvtColor(dist, dist, CV_8U);
	std::cout << imgRoi.type() << " " << dist8u.type() << std::endl;
	//cv::Canny(dist8u, dist8u, 0, 240);
	std::vector<cv::Vec4i> lines;
	std::vector<cv::Vec4i> flines;
	cv::HoughLinesP(dist8u, lines, 1,  CV_PI / 180, 20, 20);
	std::cout << lines.size();
	cv::cvtColor(dist8u, dist8u, cv::COLOR_GRAY2BGR);
	for (size_t i = 1; i < lines.size(); i++)
	{
		cv::Vec4i l = lines[i];

		
		//double angle = atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;

		/*if (angle < 150 && angle >= 30) {
			flines.push_back(lines[i]);
			std::cout << l[0] << "," << l[1] << "," << l[2] << "," << l[3] << std::endl;
			
		}*/
		cv::line(dist8u, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 1);
	}

	print(dist8u, "dist");
	imgRoi=Rotation(rotImg, cv::Point(lines[0][0], lines[0][1]), cv::Point(lines[0][2], lines[0][3]));
	print(rotImg, "imgROI");
	print(imgRoi, "Result");




	//cv::Mat marker(cv::Size(img.cols, img.rows), img.type());
	//for (int j = 0; j < 4; j++)
	//{
	//	line(marker, rect_points[j], rect_points[(j + 1) % 4], color, 3);
	//}

	//cv::Mat hlines;
	//cv::cvtColor(imgRoi, hlines, cv::COLOR_BGR2GRAY);
	////hlines = sharpering(hlines);
	////cv::pyrUp(hlines, hlines, cv::Size(hlines.cols*2, hlines.rows*2));
	//cv::threshold(hlines, hlines, 180, 255, cv::THRESH_OTSU);
	//
	//cv::Canny(hlines, hlines, 150, 300, 5);
	//std::vector<std::vector<cv::Point> > conr;
	//cv::findContours(hlines, conr, cv::RETR_CCOMP, cv::CHAIN_APPROX_TC89_KCOS, cv::Point());
	//cv::cvtColor(hlines, hlines, cv::COLOR_GRAY2BGR);
	/*for (int i = 0; i < conr.size(); i++)
	{
		cv::drawContours(hlines, conr, (int)i, cv::Scalar(rng(255), rng(255), rng(255)),-1);
	}*/

	

	
	//imgRoi = Rotation(imgRoi, cv::Point(flines[0][2], flines[0][3]), cv::Point(flines[0][0], flines[0][1]));
	//marker = Rotation(marker, cv::Point(flines[0][2], flines[0][3]), cv::Point(flines[0][0], flines[0][1]));
	//print(imgRoi, "Hough");
	cv::waitKey(0);
	return 0;
}
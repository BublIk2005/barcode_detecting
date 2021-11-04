#include<opencv2/opencv.hpp>
#include<algorithm>
#include <cstdlib>
#include<math.h>
#include<utility>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

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
	cv::Point2f vec = b - a;
	cv::Point2f dir(0, img.rows);
	float cos = (vec.x * dir.x + vec.y * dir.y) / (sqrt(vec.x * vec.x + vec.y * vec.y) * sqrt(dir.x * dir.x + dir.y * dir.y));
	float angle = -57.3*acosf(cos);

	cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
	cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
	cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), img.size(), angle).boundingRect2f();
	rot.at<double>(0, 2) += bbox.width / 2.0 - img.cols / 2.0;
	rot.at<double>(1, 2) += bbox.height / 2.0 - img.rows / 2.0;

	cv::Mat result;
	cv::warpAffine(img, result, rot, bbox.size());
	
	
	
	return result;
}

cv::Mat Rotation(cv::Mat img, cv::Point2f a, cv::Point2f b, cv::Point2f* rect_points)
{
	cv::Point2f vec = b - a;
	cv::Point2f dir(0, img.rows);
	float cos = (vec.x * dir.x + vec.y * dir.y) / (sqrt(vec.x * vec.x + vec.y * vec.y) * sqrt(dir.x * dir.x + dir.y * dir.y));
	float sin = (vec.x * dir.y - vec.y * dir.x) / (sqrt(vec.x * vec.x + vec.y * vec.y) * sqrt(dir.x * dir.x + dir.y * dir.y));
	float angle = 57.3 *  acosf(cos);
	if (sin < 0 && cos < 0 || cos>0 && sin > 0)
		angle = -angle;

	cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
	cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
	cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), img.size(), angle).boundingRect2f();
	rot.at<double>(0, 2) += bbox.width / 2.0 - img.cols / 2.0;
	rot.at<double>(1, 2) += bbox.height / 2.0 - img.rows / 2.0;

	cv::Mat result;
	cv::warpAffine(img, result, rot, bbox.size());
	cv::Point2f newPoint;
	for (int i = 0; i < 4; ++i) {
		newPoint.x = (rect_points[i].x-center.x)*cos - (rect_points[i].y-center.y)*sin + center.x * (result.cols / (float)img.cols);
		newPoint.y = (rect_points[i].x - center.x) * sin + (rect_points[i].y - center.y) * cos+center.y * (result.rows / (float)img.rows);
		//newPoint = newPoint * (result.cols / (float)img.cols);
		rect_points[i] = newPoint;
	}


	return result;
}
cv::Mat Rotation(cv::Mat img, cv::RotatedRect rect, cv::Point2f* rect_points)
{
	float angle = rect.angle;
	if (abs(angle) > 45.)
		angle += 90;
	//cv::Point2f center = rect.center;
	cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
	cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
	cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), img.size(), angle).boundingRect2f();
	rot.at<double>(0, 2) += bbox.width / 2.0 - img.cols / 2.0;
	rot.at<double>(1, 2) += bbox.height / 2.0 - img.rows / 2.0;
	float cosin = cos(angle);
	float sinus = sin(angle);

	cv::Mat result;
	cv::warpAffine(img, result, rot, bbox.size());
	cv::Point2f newCenter(bbox.width/2., bbox.height / 2.0);
	cv::Point2f newPoint;
	for (int i = 0; i < 4; ++i) {
		newPoint.x = (rect_points[i].x - center.x) * cosin - (rect_points[i].y - center.y) * sqrt(1 - cosin * cosin) + center.x * (result.cols / (float)img.cols);
		newPoint.y = (rect_points[i].x - center.x) * sqrt(1 - cosin * cosin) + (rect_points[i].y - center.y) * cosin + center.y * (result.rows / (float)img.rows);
		//newPoint = newPoint * (result.cols / (float)img.cols);
		rect_points[i] = newPoint;
	}


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
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta = 0, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta = 0, cv::BORDER_DEFAULT);
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

cv::Mat skelet(cv::Mat img)
{
	cv::threshold(img, img, 127, 255, cv::THRESH_BINARY);
	cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat temp;
	cv::Mat eroded;

	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

	bool done;
	do
	{
		cv::erode(img, eroded, element);
		cv::dilate(eroded, temp, element); // temp = open(img)
		cv::subtract(img, temp, temp);
		cv::bitwise_or(skel, temp, skel);
		eroded.copyTo(img);
		done = (cv::countNonZero(img) == 0);
	} while (!done);
	return skel;
}


cv::Mat drawHistogram(const cv::Mat& imeg_bin, int intensiv) {
	std::vector<int> histogram(20, 0);
	int pixels = 0;
	for (int i = 0; i < imeg_bin.cols; i++) {
		if (int(imeg_bin.at<uchar>(imeg_bin.rows / 2, i)) == intensiv)
		{
			pixels++;
		}
		else
		{
			if (pixels != 0)
				histogram[pixels]++;
			pixels = 0;
		}
				
	}
	int hist_w = histogram.size()*2; int hist_h = histogram.size() * 2;
	cv::Mat hist(cv::Mat::zeros(hist_w, hist_h,CV_8UC1));
	for (int i = 0; i < histogram.size(); i++)
	{
		rectangle(hist, cv::Point(i * 2, hist_h-histogram[i]), cv::Point(i * 2 + 2, hist_h ), cv::Scalar(255), -1);
	}
	return hist;
}
std::vector<int> Histogram(const cv::Mat& imeg_bin, int intensiv) {
	std::vector<int> histogram(20, 0);
	int pixels = 0;
	for (int i = 0; i < imeg_bin.cols; i++) {
		if (int(imeg_bin.at<uchar>(imeg_bin.rows / 2, i)) == intensiv)
		{
			pixels++;
		}
		else
		{
			if (pixels != 0)
				histogram[pixels]++;
			pixels = 0;
		}

	}
	return histogram;
}
std::vector<std::pair<int, int>> normalizeHist(std::vector<int> hist) {
	int locMax=0;
	int locMaxInd = -1;
	std::vector<std::pair<int,int>> normalizeVec;
	std::pair<int, int> newValue;
	std::vector<int> locMaxVec;
	for (int i = 0; i < hist.size(); i++)
	{
		if (hist[i] > 0 && hist[i]>locMax) {
			locMax = hist[i];
			locMaxInd = i;
		}
		else if (locMaxInd != -1) {
			locMaxVec.push_back(locMaxInd);
			locMax = 0;
			locMaxInd = -1;
		}
	}
	int ind = 0;
	int flag = 0;
	for (int i = 0; i < hist.size(); i++)
	{
		
		if (hist[i] > 0) {
			newValue.first = i;
			newValue.second = locMaxVec[ind];
			normalizeVec.push_back(newValue);
			flag = 1;
		}
		else if(flag==1) {
			++ind;
			flag = 0;
		}
	}
	return normalizeVec;
}

cv::Mat drawCode(cv::Mat& imeg_bin, std::vector<int> Bhist, std::vector<int> Whist)
{
	
	int bpixels = 0;
	int wpixels = 0;
	std::vector<int> Blck;
	std::vector<int> Whit;
	std::vector<std::pair<int, int>> newBhist = normalizeHist(Bhist);
	std::vector<std::pair<int, int>> newWhist = normalizeHist(Whist);
	
	
	for (int i = 0; i < imeg_bin.cols; i++) {
		int j = 0;
		if (int(imeg_bin.at<uchar>(imeg_bin.rows / 2, i)) == 0)
		{
			if (wpixels != 0)
			{
				for (int i = 0; i < newWhist.size(); i++)
				{
					if (wpixels == newWhist[i].first)
						wpixels = newWhist[i].second;
				}
				Whit.push_back(wpixels);
			}
			wpixels = 0;
			bpixels++;
		}
		else
		{
			if (bpixels != 0) {
				for (int i = 0; i < newBhist.size(); i++)
				{
					if (bpixels == newBhist[i].first)
						bpixels = newBhist[i].second;
				}
				Blck.push_back(bpixels);
			}
			bpixels = 0;
			wpixels++;
		}
	}
	int mat_h, mat_w;
	mat_h = 50;
	mat_w = 0;
	for (auto item : Blck) {
		std::cout << item << std::endl;
		mat_w += item;
	}
	for (auto item : Whit) {
		mat_w += item;
	}
	//mat_w += 200;
	cv::Mat barcode(mat_h, mat_w, CV_8UC1);
	int step = 0;
	if (int(imeg_bin.at<uchar>(imeg_bin.rows / 2, 0)) == 0)
	{
		for (int i = 0; i < Blck.size(); ++i) {
			rectangle(barcode, cv::Point(step, 0), cv::Point(step + Blck[i], mat_h), cv::Scalar(0), -1);
			rectangle(barcode, cv::Point(step + Blck[i], 0), cv::Point(step + Blck[i] + Whit[i], mat_h), cv::Scalar(255), -1);
			step = step + Blck[i] + Whit[i];
		}
	}
	else
	{
		for (int i = 0; i < Whit.size(); ++i) {
			rectangle(barcode, cv::Point(step, 0), cv::Point(step + Whit[i], mat_h), cv::Scalar(255), -1);
			rectangle(barcode, cv::Point(step + Whit[i], 0), cv::Point(step + Blck[i] + Whit[i], mat_h), cv::Scalar(0), -1);
			step = step + Blck[i] + Whit[i];
		}
	}
	
	print(barcode, "NormalizeBarCode");
	return barcode;
}
int main()
{
	cv::Point anchor = cv::Point(-1, -1);
	double delta = 0;
	int ddepth = -1;
	cv::Mat kernel;
	kernel = cv::Mat::ones(3, 3, CV_32F) / (float)(3 * 3);

	cv::Mat img = cv::imread("E:/Projects/barcode_detecting/data/test3.jpg", cv::IMREAD_COLOR);
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
	cv::Mat kernel_grad, blured_grad;
	kernel_grad = cv::Mat::ones(3, 3, CV_32F);
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta=1, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta=1, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	
	cv::subtract(abs_grad_x, abs_grad_y, grad);
	//grad = grad_x*0.5 +  grad_y*0.5;
	cv::convertScaleAbs(grad, grad);
	
	//print(abs_grad_y, "grad_y");
	//print(abs_grad_x, "grad_x");
	//print(grad, "grad");
	
	cv::threshold(grad, grad, 200, 255, cv::THRESH_OTSU);
	cv::erode(grad, grad, cv::Mat::ones(1, 1, CV_32F), cv::Point(-1, -1), 1);
	cv::dilate(grad, grad, cv::Mat::ones(1, 1, CV_32F), cv::Point(-1, -1), 2);
	cv::erode(grad, grad, cv::Mat::ones(1, 1, CV_32F), cv::Point(-1, -1), 1);
	cv::GaussianBlur(grad, grad, cv::Size(9, 9), 1, 0, cv::BORDER_DEFAULT);
	cv::filter2D(grad, blured_grad, CV_8UC1, kernel_grad);
	//print(blured_grad, "blured_grad");
	//blured_grad = grad;
	cv::Mat blured, blured_bin;
	//cv::GaussianBlur(blured_grad, blured, cv::Size(5, 5), 1, 0, cv::BORDER_DEFAULT);
	//cv::blur(grad, blured, cv::Size_<int>(9, 9));
	blured = blured_grad;
	//print(blured, "blur");
	
	cv::threshold(blured, blured_bin, 100, 255, cv::THRESH_OTSU);
	//blured_bin = blured_grad;
	/*cv::erode(blured_bin, blured_bin, cv::Mat::ones(1, 1, CV_32F), cv::Point(-1, -1), 1);
	cv::dilate(blured_bin, blured_bin, cv::Mat::ones(1, 1, CV_32F), cv::Point(-1, -1), 2);
	cv::erode(blured_bin, blured_bin, cv::Mat::ones(1, 1, CV_32F), cv::Point(-1, -1), 1);*/

	
	/*print(blured_bin, "erodil");*/

	kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size_<int>(img.cols/100, 2));
	cv::morphologyEx(blured_bin, blured_bin, cv::MORPH_CLOSE, kernel);
	
	//print(blured_bin, "blured_bin");
	//cv::Canny(blured_bin, blured_bin, 50, 200);
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hireachy;
	cv::findContours(blured_bin, contours, hireachy,cv::RETR_CCOMP,cv::CHAIN_APPROX_TC89_KCOS, cv::Point());
	cv::Mat drawing = cv::Mat::zeros(blured_bin.size(), CV_8UC3);
	int ncomp = contours.size();
	
	contours=sortContours(contours);
	
	
	
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
				if (area > 0.5)
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
	cv::copyTo(img, contourIm, img);
	/*for (int i = 0; i < ncomp; i++)
	{
		cv::drawContours(contourIm, contours, (int)i, cv::Scalar(rng(255), rng(255), rng(255)), 3);
	}*/
	
	//contourIm = img;
	cv::Rect cont(rect_points[0], rect_points[1]);
	
		for (int j = 0; j < 4; j++)
		{
			line(contourIm, rect_points[j], rect_points[(j + 1) % 4], color,3);
		}
		cv::Point a, b; 
		if (lineLenght(rect_points[0], rect_points[1]) > lineLenght(rect_points[1], rect_points[2]))
		{
			a = (rect_points[0] + rect_points[1]) / 2;
			b = (rect_points[2] + rect_points[3]) / 2;
		}
		else {
			a = (rect_points[1] + rect_points[2]) / 2;
			b = (rect_points[3] + rect_points[1]) / 2;
		}

		if (a.y > b.y)
		{
			std::swap(a, b);
		}
		
		cv::line(contourIm, a, b, cv::Scalar(255, 0, 0), 1);
		

	print(contourIm, "contours");
	cv::Mat rot, rotImg;
	cv::copyTo(blured_bin, rot, blured_bin);
	rot = Rotation(img, a,b, rect_points);
	/*for (int j = 0; j < 4; j++)
	{
		std::cout << rect_points[j];
		line(rot, rect_points[j], rect_points[(j + 1) % 4], color, 3);
	}*/
	//print(rot, "Rotation1");

	cv::RotatedRect rotRect(rect_points[0], rect_points[1], rect_points[2]);
	cv::Rect roi = rotRect.boundingRect2f();
	cv::Mat imgRoi = rot(roi);
	print(imgRoi, "ROI");
	//imgRoi = sharpering(imgRoi);
	print(imgRoi, "ROI1");
	if (imgRoi.cols / 90 < 3)
	{
		cv::pyrUp(imgRoi, imgRoi, cv::Size(imgRoi.cols*2, imgRoi.rows*2), cv::BORDER_DEFAULT);
	}
	std::cout << imgRoi.cols << std::endl;
	cv::cvtColor(imgRoi, imgRoi, cv::COLOR_BGR2GRAY);
	cv::threshold(imgRoi, imgRoi, 220, 255, cv::THRESH_OTSU);
	print(imgRoi, "ROI_BINARY");
	
	std::vector<int> Whist=Histogram(imgRoi, 255);
	std::vector<int> Bhist =Histogram(imgRoi, 0);
	std::vector<std::pair<int, int>> W = normalizeHist(Whist);
	
	for (auto& item:W)
	{
		std::cout << item.first << " " << item.second << std::endl;
	}

	drawCode(imgRoi, Bhist, Whist);
	
	//std::vector<cv::Vec4i> lines;
	//std::vector<cv::Vec4i> flines;
	//cv::HoughLinesP(imgRoi, lines, 1,  CV_PI / 180, 100, 20);
	//std::cout << lines.size();
	//cv::cvtColor(imgRoi, imgRoi, cv::COLOR_GRAY2BGR);
	//for (size_t i = 1; i < lines.size(); i++)
	//{
	//	cv::Vec4i l = lines[i];

	//	
	//	double angle = atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;

	//	if (angle < 110 && angle >= 80) {
	//		flines.push_back(lines[i]);
	//		std::cout << l[0] << "," << l[1] << "," << l[2] << "," << l[3] << std::endl;
	//		
	//	}
	//	cv::line(imgRoi, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 0, 255), 1);
	//}
	//print(imgRoi, "ROI_Lines");



	/*minRect.points(rect_points);
	rotImg=Rotation(img, a,b, rect_points);
	for (int j = 0; j < 4; j++)
	{
		std::cout << rect_points[j];
		line(rotImg, rect_points[j], rect_points[(j + 1) % 4], color, 3);
	}
	print(rotImg, "Rotation2");*/
	//cv::Rect2f bbox = minRect;
	//cv::Rect roi(rect_points[2], rect_points[3]);
	
	/*cv::Rect roi = minRect.boundingRect2f();
	cv::Mat imgRoi = rot(roi);
	rotImg = img(roi);
	print(imgRoi, "ROI1");*/
	

	//cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(imgRoi.cols/10, 1));
	//cv::morphologyEx(imgRoi, imgRoi, cv::MORPH_CLOSE, element);
	//
	//print(imgRoi, "ROI");
	//cv::Mat sk;
	//imgRoi= skelet(imgRoi);
	//print(imgRoi, "skelet");
	//
	////
	//std::vector<cv::Vec4i> lines;
	//std::vector<cv::Vec4i> flines;
	//cv::HoughLinesP(imgRoi, lines, 1,  CV_PI / 180, 20, 20);
	//std::cout << lines.size();
	//cv::cvtColor(imgRoi, imgRoi, cv::COLOR_GRAY2BGR);
	//for (size_t i = 1; i < lines.size(); i++)
	//{
	//	cv::Vec4i l = lines[i];

	//	
	//	//double angle = atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;

	//	/*if (angle < 150 && angle >= 30) {
	//		flines.push_back(lines[i]);
	//		std::cout << l[0] << "," << l[1] << "," << l[2] << "," << l[3] << std::endl;
	//		
	//	}*/
	//	cv::line(imgRoi, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 0, 255), 1);
	//}
	//
	//print(imgRoi, "dist");
	////imgRoi=Rotation(rotImg, cv::Point(lines[0][0], lines[0][1]), cv::Point(lines[0][2], lines[0][3]));
	////imgRoi = Rotation(rotImg, cv::Point(lines[0][0], lines[0][1]), cv::Point(lines[0][2], lines[0][3]));
	//print(rotImg, "imgROI");
	//print(imgRoi, "Result");




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
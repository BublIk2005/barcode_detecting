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
	cv::Mat kernel = (cv::Mat_<double>(3, 3) << 0, -1/4, 0, 1/4, 0, -1/4, 0, -1/4, 0); 
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
cv::Mat scaler(cv::Mat& src, double &scale)
{
	cv::Mat res;
	double model = 600.;
	if (src.cols > model) {
		scale = src.cols /model;
		cv::resize(src, res, cv::Size(int(src.cols / scale), int(src.rows / scale)), 0, 0, cv::INTER_AREA);
	}
	else if (model > src.cols) {
		scale = model / src.cols;
		cv::resize(src, res, cv::Size(int(src.cols * scale), int(src.rows * scale)), 0, 0, cv::INTER_AREA);
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
		cv::dilate(eroded, temp, element); 
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
std::vector<int> countStrokes(const cv::Mat& imeg_bin, int intensiv) {
	std::vector<int> count;
	int pixels = 0;
	for (int i = 0; i < imeg_bin.cols; i++) {
		if (int(imeg_bin.at<uchar>(imeg_bin.rows / 2, i)) == intensiv)
		{
			pixels++;
		}
		else
		{
			if (pixels != 0)
				count.push_back(pixels);
			pixels = 0;
		}
	}
	if (intensiv == 255 && int(imeg_bin.at<uchar>(imeg_bin.rows / 2, 0)) == intensiv)
	{
		count.erase(count.begin());
	}
	return count;
}
std::vector<int> normalizeVec(std::vector<int> histB, std::vector<int> histW) {
	std::vector<int> normalizeVec(histB.size() + histW.size(), 0);
	int zeroMod = histB[0];
	int j = 0;
	for (size_t i = 0; i < normalizeVec.size()-1; i+=2)
	{
		normalizeVec[i] =round(histB[j] / (double)zeroMod); 
		
		normalizeVec[i + 1] = round(histW[j] / (double)zeroMod);
		
		j++;
	}
	normalizeVec[normalizeVec.size() - 1] = round(histB[histB.size() - 1] / (double)zeroMod);
	return normalizeVec;
}
std::vector<int> normalizeVecBit(std::vector<int> histB, std::vector<int> histW) {
	std::vector<int> normalizeVec;
	int sizeV = histB.size() + histW.size();
	int zeroMod = histB[0];
	int j = 0;
	for (int i = 0; i < histB.size() + histW.size() - 1; i += 2)
	{
		for (int i = 0; i < round(histB[j] / (double)zeroMod); i++)
		{
			normalizeVec.push_back(1);
		}
		for (int i = 0; i < round(histW[j] / (double)zeroMod); i++)
		{
			normalizeVec.push_back(0);
		}
		j++;
	}
	normalizeVec[normalizeVec.size() - 1] = round(histB[histB.size() - 1] / (double)zeroMod);
	return normalizeVec;
}

//cv::Mat drawNormalizeCode(cv::Mat& imeg_bin, std::vector<int> Bhist, std::vector<int> Whist)
//{
//	
//	int bpixels = 0;
//	int wpixels = 0;
//	std::vector<int> Blck;
//	std::vector<int> Whit;
//	std::vector<std::pair<int, int>> newBhist = normalizeHist(Bhist);
//	std::vector<std::pair<int, int>> newWhist = normalizeHist(Whist);
//	
//	
//	for (int i = 0; i < imeg_bin.cols; i++) {
//		int j = 0;
//		if (int(imeg_bin.at<uchar>(imeg_bin.rows / 2, i)) == 0)
//		{
//			if (wpixels != 0)
//			{
//				for (int i = 0; i < newWhist.size(); i++)
//				{
//					if (wpixels == newWhist[i].first)
//						wpixels = newWhist[i].first;
//				}
//				Whit.push_back(wpixels);
//			}
//			wpixels = 0;
//			bpixels++;
//		}
//		else
//		{
//			if (bpixels != 0) {
//				for (int i = 0; i < newBhist.size(); i++)
//				{
//					if (bpixels == newBhist[i].first)
//						bpixels = newBhist[i].first;
//				}
//				Blck.push_back(bpixels);
//			}
//			bpixels = 0;
//			wpixels++;
//		}
//	}
//	int mat_h, mat_w;
//	mat_h = 50;
//	mat_w = 0;
//	for (auto item : Blck) {
//		std::cout << item << std::endl;
//		mat_w += item;
//	}
//	for (auto item : Whit) {
//		mat_w += item;
//	}
//	//mat_w += 200;
//	cv::Mat barcode(mat_h, mat_w, CV_8UC1);
//	int step = 0;
//	if (int(imeg_bin.at<uchar>(imeg_bin.rows / 2, 0)) == 0)
//	{
//		for (int i = 0; i < Blck.size(); ++i) {
//			rectangle(barcode, cv::Point(step, 0), cv::Point(step + Blck[i], mat_h), cv::Scalar(0), -1);
//			rectangle(barcode, cv::Point(step + Blck[i], 0), cv::Point(step + Blck[i] + Whit[i], mat_h), cv::Scalar(255), -1);
//			step = step + Blck[i] + Whit[i];
//		}
//	}
//	else
//	{
//		for (int i = 0; i < Whit.size(); ++i) {
//			rectangle(barcode, cv::Point(step, 0), cv::Point(step + Whit[i], mat_h), cv::Scalar(255), -1);
//			rectangle(barcode, cv::Point(step + Whit[i], 0), cv::Point(step + Blck[i] + Whit[i], mat_h), cv::Scalar(0), -1);
//			step = step + Blck[i] + Whit[i];
//		}
//	}
//	
//	print(barcode, "NormalizeBarCode");
//	return barcode;
//}
cv::Mat drawCode(cv::Mat& imeg_bin, std::vector<int> normalizeVec)
{
	
	
	int mat_h, mat_w;
	mat_h = 50;
	mat_w = 0;
	for (auto item : normalizeVec) {
		
		mat_w += item;
	}
	cv::Mat barcode(mat_h, mat_w, CV_8UC1);
	int step = 0;
		for (int i = 0; i < normalizeVec.size()-1; i+=2) {
			rectangle(barcode, cv::Point(step, 0), cv::Point(step + normalizeVec[i], mat_h), cv::Scalar(0), -1);
			rectangle(barcode, cv::Point(step + normalizeVec[i], 0), cv::Point(step + normalizeVec[i] + normalizeVec[i+1], mat_h), cv::Scalar(255), -1);
			step = step + normalizeVec[i] + normalizeVec[i+1];
		}
		rectangle(barcode, cv::Point(step, 0), cv::Point(step + normalizeVec[normalizeVec.size()-1], mat_h), cv::Scalar(0), -1);
	print(barcode, "NormalizeBarCode");
	return barcode;
}

std::vector<int> decoder(std::vector<int> vecBit)
{
	std::vector<int> result(13, 0);
	std::string firstNum("");
	std::map<std::string, std::pair<int, std::string>> encodingNumTable = {
		{"0001101",{0,"L"}},
		{"1110010",{0,"R"}},
		{"0100111",{0,"G"}},
		{"0011001",{1,"L"}},
		{"1100110",{1,"R"}},
		{"0110011",{1,"G"}},
		{"0010011",{2,"L"}},
		{"1101100",{2,"R"}},
		{"0011011",{2,"G"}},
		{"0111101",{3,"L"}},
		{"1000010",{3,"R"}},
		{"0100001",{3,"G"}},
		{"0100011",{4,"L"}},
		{"1011100",{4,"R"}},
		{"0011101",{4,"G"}},
		{"0110001",{5,"L"}},
		{"1001110",{5,"R"}},
		{"0111001",{5,"G"}},
		{"0101111",{6,"L"}},
		{"1010000",{6,"R"}},
		{"0000101",{6,"G"}},
		{"0111011",{7,"L"}},
		{"1000100",{7,"R"}},
		{"0010001",{7,"G"}},
		{"0110111",{8,"L"}},
		{"1001000",{8,"R"}},
		{"0001001",{8,"G"}},
		{"0001011",{9,"L"}},
		{"1110100",{9,"R"}},
		{"0010111",{9,"G"}}
	};
	std::map<std::string, int> encodingFirstNumTable = {
		{"LLLLLLRRRRRR",0},
		{"LLGLGGRRRRRR",1},
		{"LLGGLGRRRRRR",2},
		{"LLGGGLRRRRRR",3},
		{"LGLLGGRRRRRR",4},
		{"LGGLLGRRRRRR",5},
		{"LGGGLLRRRRRR",6},
		{"LGLGLGRRRRRR",7},
		{"LGLGGLRRRRRR",8},
		{"LGGLGLRRRRRR",9}
	};
	int k = 1;
	for (int i(3); i < 44; i+=7)
	{
		std::string encod = "";
		for (int j = i; j - i < 7; ++j) {
			encod = encod + std::to_string(vecBit[j]);
		}
		std::pair<int, std::string> decod = encodingNumTable.find(encod)->second;
		result[k] = decod.first;
		firstNum += decod.second;
		k++;
	}
	for (int i(50); i < 92; i += 7)
	{
		std::string encod = "";
		for (int j = i; j - i < 7; ++j) {
			encod = encod + std::to_string(vecBit[j]);
		}
		std::pair<int, std::string> decod = encodingNumTable.find(encod)->second;
		result[k] = decod.first;
		firstNum += decod.second;
		k++;
	}
	result[0] = encodingFirstNumTable.find(firstNum)->second;
	return result;
}
int main()
{
	cv::Point anchor = cv::Point(-1, -1);
	double delta = 0;
	int ddepth = -1;
	cv::Mat kernel;
	kernel = cv::Mat::ones(3, 3, CV_32F) / (float)(3 * 3);

	cv::Mat img = cv::imread("E:/Projects/barcode_detecting/data/test4.jpg", cv::IMREAD_COLOR);
	cv::Mat src_gray, src_bin, src_fil;
	cv::cvtColor(img, src_gray, cv::COLOR_BGR2GRAY);
	src_fil = filtred(src_gray);
	//cv::Mat_<double> src_sh = src_gray;
	//src_sh = sharpering(src_sh);
	//print(src_sh, "filt");
	//cv::threshold(src_sh, src_bin, 128, 255, cv::THRESH_OTSU);
	cv::RNG rng(12345);
	print(img, "barcode");
	
	//src_gray=scalePyr(src_gray);
	double sc = 0;
	src_gray = scaler(src_gray,sc);
	
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
	kernel_grad = cv::Mat::ones(2, 9, CV_32F);
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
	//cv::GaussianBlur(grad, grad, cv::Size(9, 9), 1, 0, cv::BORDER_DEFAULT);
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

	/*kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size_<int>(1, 3));
	cv::morphologyEx(blured_bin, blured_bin, cv::MORPH_CLOSE, kernel);*/
	
	print(blured_bin, "blured_bin");
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
	if (img.cols > 600) {
		for (size_t i = 0; i < 4; i++)
		{
			rect_points[i] = rect_points[i] * sc;
		}
	}
	else if (600 > img.cols) {
		for (size_t i = 0; i < 4; i++)
		{
			rect_points[i] = rect_points[i] / sc;
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
	//cv::imshow("RealR", imgRoi);
	print(imgRoi, "ROI");
	//imgRoi = sharpering(imgRoi);
	int aP = 0;
	int bP = 0;
	
	//imgRoi=Rotation(imgRoi, P1, P2);
	if (imgRoi.cols / 90 < 3)
	{
		cv::pyrUp(imgRoi, imgRoi, cv::Size(imgRoi.cols*2, imgRoi.rows*2), cv::BORDER_DEFAULT);
	}
	print(imgRoi, "ROI1");
	//cv::Mat roiBin = imgRoi;
	cv::cvtColor(imgRoi, imgRoi, cv::COLOR_BGR2GRAY);
	//cv::Canny(imgRoi, imgRoi, 80, 255);
	print(imgRoi, "ROI_GRAY");
	cv::threshold(imgRoi, imgRoi, 100, 255, cv::THRESH_OTSU);
	
	/*while (int(roiBin.at<uchar>(roiBin.rows / 2, aP)) == 255)
	{
		aP++;
		std::cout << "lox" << aP << std::endl;
	}
	while (int(roiBin.at<uchar>(roiBin.rows / 4,bP)) == 255)
	{
		bP++;
	}
	cv::Point2f P1(bP, roiBin.rows / 4);
	cv::Point2f P2(aP, roiBin.rows / 2);*/
	/*cv::Point2f P1(0, imgRoi.rows/2);
	cv::Point2f P2(imgRoi.cols , imgRoi.rows / 2);*/
	//cv::line(imgRoi, P1, P2, cv::Scalar(100, 100, 100));
	//imgRoi=Rotation(imgRoi, P1, P2);
	print(imgRoi, "ROI_BIN");
	
	std::vector<int> Whist=countStrokes(imgRoi, 255);
	std::vector<int> Bhist = countStrokes(imgRoi, 0);
	std::vector<int> vec = normalizeVec(Bhist,Whist);
	std::vector<int> vecBit = normalizeVecBit(Bhist, Whist);
	/*for (auto it : vecBit)
	{
		std::cout << it;
	}*/
	
	drawCode(imgRoi, vec);
	std::vector<int> res = decoder(vecBit);
	for (auto item : res) {
		std::cout << item;
	}
	cv::waitKey(0);
	return 0;
}
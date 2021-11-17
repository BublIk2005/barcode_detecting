#include"barcode_detecting.h"


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

//600x450
cv::Mat scaler(cv::Mat& src, double &scale, double modelSize)
{
	cv::Mat res;
	double model = modelSize;
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
void contoursScaler(cv::Mat img, cv::Point2f* rect_points, double& scale, double modelSize)
{
	if (img.cols > modelSize) {
		for (size_t i = 0; i < 4; i++)
		{
			rect_points[i] = rect_points[i] * scale;
		}
	}
	else if (modelSize > img.cols) {
		for (size_t i = 0; i < 4; i++)
		{
			rect_points[i] = rect_points[i] / scale;
		}
	}
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
		
		rect_points[i] = newPoint;
	}


	return result;
}
cv::Mat Rotation(cv::Mat img, cv::RotatedRect rect, cv::Point2f* rect_points)
{
	float angle = rect.angle;
	if (abs(angle) > 45.)
		angle += 90;
	
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
		
		rect_points[i] = newPoint;
	}


	return result;
}


cv::Mat Gradient(cv::Mat src_gray)
{
	cv::Mat grad_x, grad_y, grad;
	cv::Mat abs_grad_x, abs_grad_y;
	int scale = 1;
	int ddepth = -1;
	double delta = 1;
	cv::Mat kernel_grad, blured_grad;
	kernel_grad = cv::Mat::ones(2, 9, CV_32F);
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta = 1, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta = 1, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	cv::subtract(abs_grad_x, abs_grad_y, grad);
	cv::convertScaleAbs(grad, grad);
	cv::threshold(grad, grad, 200, 255, cv::THRESH_OTSU);
	cv::erode(grad, grad, cv::Mat::ones(1, 1, CV_32F), cv::Point(-1, -1), 1);
	cv::dilate(grad, grad, cv::Mat::ones(1, 1, CV_32F), cv::Point(-1, -1), 2);
	cv::erode(grad, grad, cv::Mat::ones(1, 1, CV_32F), cv::Point(-1, -1), 1);
	cv::filter2D(grad, grad, CV_8UC1, kernel_grad);
	cv::threshold(grad, grad, 100, 255, cv::THRESH_OTSU);
	print(grad, "blured_bin");
	return grad;
}
void contours(cv::Mat & grad, cv::Point2f* rect_points) {
	cv::Mat blured_bin = grad;
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hireachy;
	cv::findContours(blured_bin, contours, hireachy, cv::RETR_CCOMP, cv::CHAIN_APPROX_TC89_KCOS, cv::Point());
	cv::Mat drawing = cv::Mat::zeros(blured_bin.size(), CV_8UC3);
	int ncomp = contours.size();
	contours = sortContours(contours);
	cv::RotatedRect minRect;
	cv::Scalar color = cv::Scalar(0, 255, 0);
	int imgArea = grad.cols * grad.rows;
	int cond = 0;
	int ind = ncomp - 1;
	while (cond < 4)
	{
		cond = 0;
		minRect = minAreaRect(contours[ind]);
		minRect.points(rect_points);
		double area = cv::contourArea(contours[ind]) / (lineLenght(rect_points[0], rect_points[1]) * lineLenght(rect_points[1], rect_points[2]));
		for (size_t i = 0; i < 4; i++)
		{

			if (rect_points[i].x < (double)blured_bin.cols && rect_points[i].y < (double)blured_bin.rows && rect_points[i].x >= 0. && rect_points[i].y >= 0.)
				if (area > 0.5)
				{
					++cond;
				}
		}
		--ind;
		if (ind == 0)
			cond = 4;
	}
}

bool check(std::vector<int> vecBit)
{
	if (vecBit.size() == 94)
		return true;
	else return false;
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
//	normalizeVec[normalizeVec.size() - 1] = round(histB[histB.size() - 1] / (double)zeroMod);
	return normalizeVec;
}

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
	if (check(vecBit) == false)
	{
		std::cout << "Failed to recognize the barcode" << std::endl;
		std::exit(0);
	}
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
void findBarcode(cv::Mat src, cv::Point2f* rect_points, double scale)
{
	cv::Mat img = src;
	cv::Mat src_gray, src_bin;
	cv::cvtColor(img, src_gray, cv::COLOR_BGR2GRAY);
	print(img, "barcode");
	double sc = 0;
	src_gray = scaler(src_gray, sc, scale);
	cv::Mat grad = Gradient(src_gray);
	contours(grad, rect_points);
	contoursScaler(img, rect_points, sc, scale);
}

int main()
{
	cv::Mat img = cv::imread("E:/Projects/barcode_detecting/data/test1.jpg", cv::IMREAD_COLOR);
	std::vector<int> standartScales = { 600, 400, 200, 100, 1200 };
	int flag=0;
	cv::Point2f rect_points[4];
	cv::Mat contourIm;
	cv::Scalar color = cv::Scalar(0, 255, 0);
	cv::Point a, b;
	std::vector<int> vecBit;
	
	cv::Mat rot, rotImg;
	cv::RotatedRect rotRect;
	cv::Rect roi;
	cv::Mat imgRoi;
	std::vector<int> Whist;
	std::vector<int> Bhist;
	std::vector<int> vec;
	bool stop = false;

	while (!stop)
	{
		findBarcode(img, rect_points, standartScales[flag]);
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
		rot = Rotation(img, a, b, rect_points);
		rotRect= cv::RotatedRect(rect_points[0], rect_points[1], rect_points[2]);
		roi = rotRect.boundingRect2f();
		imgRoi = rot(roi);
		cv::cvtColor(imgRoi, imgRoi, cv::COLOR_BGR2GRAY);
		cv::threshold(imgRoi, imgRoi, 100, 255, cv::THRESH_OTSU);
		Whist = countStrokes(imgRoi, 255);
		Bhist = countStrokes(imgRoi, 0);
		vec = normalizeVec(Bhist, Whist);
		vecBit = normalizeVecBit(Bhist, Whist);
		++flag;
		if ((vecBit.size() == 94) || (flag == standartScales.size()-1))
		{
			stop = true;
		}
	} 



	cv::copyTo(img, contourIm, img);
	for (int j = 0; j < 4; j++)
	{
		line(contourIm, rect_points[j], rect_points[(j + 1) % 4], color, 3);
	}
	print(contourIm, "contours");
	print(imgRoi, "ROI");
	print(imgRoi, "ROI_BIN");
	drawCode(imgRoi, vec);
	std::vector<int> res = decoder(vecBit);
	for (auto item : res) {
		std::cout << item;
	}
	cv::waitKey(0);
	return 0;
}
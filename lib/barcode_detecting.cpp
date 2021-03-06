#include"barcode_detecting.h"


void print(const cv::Mat &src, std::string window_name)
{
	cv::namedWindow(window_name, cv::WINDOW_NORMAL);//??????? ???? ? ?????? ??????? ???????????
	cv::resizeWindow(window_name, 600, 400);// ???????? ??????????? ?????? ???? ??? ???????? ?????????? ?????????????
	imshow(window_name, src); //??????? ??????????? ?? ?????
}
cv::Mat sharp(const cv::Mat& src)
{
	cv::Mat result;
	cv::Mat kernel = (cv::Mat_<int>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1);
	filter2D(src, result, -1, kernel);
	return result;
}

cv::Mat scaler(const cv::Mat& src, double &scale, double modelSize)
{
	//????????? ?????? ???????
	cv::Mat res;
	//?????? ??????????? ????? ???????????????
	double model = modelSize;

	if (src.cols > model) {//???? ???????? ??????????? ?????? ??????????
		scale = src.cols /model;//??????? ??????? ?????? ??????????? ???????????? ?????????
		cv::resize(src, res, cv::Size(int(src.cols / scale), int(src.rows / scale)), 0, 0, cv::INTER_AREA);//???????? ??????? ???????????
	}
	else if (model > src.cols) {//???? ???????? ??????????? ?????? ??????????
		scale = model / src.cols;//??????? ??????? ?????? ??????????? ???????????? ?????????
		cv::resize(src, res, cv::Size(int(src.cols * scale), int(src.rows * scale)), 0, 0, cv::INTER_AREA);//???????? ??????? ???????????
	}
	else
		res = src;//???? ??????????? ??? ????? ?????? ?????? ?? ?????? ?? ??????
	return res;
}
void contoursScaler(const cv::Mat &img, cv::Point2f* rect_points, double scale, double modelSize)
{
	//?????????? ???????? ?????? ??????????? ? ???????? ?? ??????? ?? ????? ????????? ??????
	//? ??????????? ? ???????? ?? ??????? ?? ????? ?????
	//?????? ?????????? ?????? ??????? ??? ????? ???????
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

cv::Mat Rotation(const cv::Mat &img, cv::Point2f a, cv::Point2f b)//?????? ??????????? ?????????? ??????????? ???? ?????: ?????? ???? ??????????? ? ??????? ????????? ????? ???????
{
	//?????????? ??????? ? ??????? ???????? ????? ???????????? ???????
	cv::Point2f vec = b - a;
	//????? ???? ???????????
	cv::Point2f dir(0, img.rows);
	//??????? ???? ????????
	float cos = (vec.x * dir.x + vec.y * dir.y) / (sqrt(vec.x * vec.x + vec.y * vec.y) * sqrt(dir.x * dir.x + dir.y * dir.y));
	//????? ???? ????????
	float sin = (vec.x * dir.y - vec.y * dir.x) / (sqrt(vec.x * vec.x + vec.y * vec.y) * sqrt(dir.x * dir.x + dir.y * dir.y));
	//???? ???????? ? ????????
	float angle = 57.3 * acosf(cos);
	//???????? ?? ???????????????? ???????????? ???????? ???????????? ???? ????? ? ????????? ???? ??? ????????? ???????? ?? 180 ???????? 
	if (sin < 0 && cos < 0 || cos>0 && sin > 0)
		angle = -angle;
	//????? ???????????
	cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
	//??????? ???????? 
	cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
	//????? ??????? ??? ???????
	cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), img.size(), angle).boundingRect2f();
	rot.at<double>(0, 2) += bbox.width / 2.0 - img.cols / 2.0;
	rot.at<double>(1, 2) += bbox.height / 2.0 - img.rows / 2.0;
	//????????? ?????? ???????
	cv::Mat result;
	//??????? ???????????
	cv::warpAffine(img, result, rot, bbox.size());
	return result;
}

cv::Mat Rotation(const cv::Mat &img, cv::Point2f a, cv::Point2f b, cv::Point2f* rect_points)//?????? ??????????? ?????????? ??????????? ???? ?????: ?????? ???? ??????????? ? ??????? ????????? ????? ???????
{
	//?????????? ??????? ? ??????? ???????? ????? ???????????? ???????
	cv::Point2f vec = b - a;
	//????? ???? ???????????
	cv::Point2f dir(0, img.rows);
	//??????? ???? ????????
	float cos = (vec.x * dir.x + vec.y * dir.y) / (sqrt(vec.x * vec.x + vec.y * vec.y) * sqrt(dir.x * dir.x + dir.y * dir.y));
	//????? ???? ????????
	float sin = (vec.x * dir.y - vec.y * dir.x) / (sqrt(vec.x * vec.x + vec.y * vec.y) * sqrt(dir.x * dir.x + dir.y * dir.y));
	//???? ???????? ? ????????
	float angle = 57.3 *  acosf(cos);
	//???????? ?? ???????????????? ???????????? ???????? ???????????? ???? ????? ? ????????? ???? ??? ????????? ???????? ?? 180 ???????? 
	if (sin < 0 && cos < 0 || cos>0 && sin > 0)
		angle = -angle;
	//????? ???????????
	cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
	//??????? ???????? 
	cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
	//????? ??????? ??? ???????
	cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), img.size(), angle).boundingRect2f();
	rot.at<double>(0, 2) += bbox.width / 2.0 - img.cols / 2.0;
	rot.at<double>(1, 2) += bbox.height / 2.0 - img.rows / 2.0;
	//????????? ?????? ???????
	cv::Mat result;
	//??????? ???????????
	cv::warpAffine(img, result, rot, bbox.size());
	//????? ?????????? ???????
	cv::Point2f newPoint;
	//??? ?????? ??????? ??????? ??????? ????? ?????????? ??? ??????????? ???????????
	for (int i = 0; i < 4; ++i) {
		newPoint.x = (rect_points[i].x-center.x)*cos - (rect_points[i].y-center.y)*sin + center.x * (result.cols / (float)img.cols);
		newPoint.y = (rect_points[i].x - center.x) * sin + (rect_points[i].y - center.y) * cos+center.y * (result.rows / (float)img.rows);
		
		rect_points[i] = newPoint;
	}


	return result;
}

cv::Mat Gradient(const cv::Mat &src_gray)
{
	// ???????? dx
	cv::Mat grad_x, 
		//???????? dy
		grad_y, 
		//?????? ???????? ??????????
		grad;
	cv::Mat abs_grad_x, abs_grad_y;
	int scale = 1;
	int ddepth = -1;
	double delta = 1;
	cv::Mat kernel_grad, blured_grad;
	//????????? ???????, ?????? ??????? ???????? ????? ??????? ????? ???? ???? ?????? ??? ?????????? ????? ???????? ????
	kernel_grad = cv::Mat::ones(2, 9, CV_32F);
	//? ??????? ??????? ?????? ??????? ?????????
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta = 1, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta = 1, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	
	cv::subtract(abs_grad_x, abs_grad_y, grad);
	
	cv::convertScaleAbs(grad, grad);
	
	cv::threshold(grad, grad, 200, 255, cv::THRESH_OTSU);
	
	//???????? ??????? ? ?????????? ??? ??????? ??????????? ?? ?????? ??????????
	cv::erode(grad, grad, cv::Mat::ones(1, 1, CV_32F), cv::Point(-1, -1), 1);
	cv::dilate(grad, grad, cv::Mat::ones(1, 1, CV_32F), cv::Point(-1, -1), 2);
	cv::erode(grad, grad, cv::Mat::ones(1, 1, CV_32F), cv::Point(-1, -1), 1);
	//????????? ??????????? ????????? ???????? ??? ??????????? ??????? ????
	cv::filter2D(grad, grad, CV_8UC1, kernel_grad);
	
	//cv::threshold(grad, grad, 100, 255, cv::THRESH_OTSU);
	//print(grad, "Binary");
	return grad;
}
void contours(const cv::Mat & grad, cv::Point2f* rect_points) {
	cv::Mat blured_bin = grad;
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hireachy;
	//? ??????? ?????????? ??????? ??????? ??? ??????? ?? ???????????
	cv::findContours(blured_bin, contours, hireachy, cv::RETR_CCOMP, cv::CHAIN_APPROX_TC89_KCOS, cv::Point());
	int ncomp = contours.size();
	//????????? ??????? ?? ???????
	contours = sortContours(contours);
	//????????????? ???????????? ??????? ? ??????? ????????? ????????
	cv::RotatedRect minRect;
	//???? ??? ???????? ?????????? ??????????? ???????
	int cond = 0;
	//?????? ?????????? ??????? 
	int ind = ncomp - 1;
	//???????? ??????? ?? ???????? ? ???????? ? ????????? ?? ?? ???????????? ???????
	while (cond < 4)
	{
		cond = 0;
		//??????????? ????????????? ? ??????? ????? ??????? ??????
		minRect = minAreaRect(contours[ind]);
		//??????????? ???????? ?????? ?????????????? ? ?????? rect_points
		minRect.points(rect_points);
		//??????? ????????? ??????? ??????? ? ??????? ?????????????? ? ??????? ?????? ??????
		double area = cv::contourArea(contours[ind]) / (lineLenght(rect_points[0], rect_points[1]) * lineLenght(rect_points[1], rect_points[2]));
		
		for (size_t i = 0; i < 4; i++)
		{
			//??? ?????? ????? ?????????????? ????????? ????? ??? ?? ??????? ?? ??????? ???????????
			if (rect_points[i].x < (double)blured_bin.cols && rect_points[i].y < (double)blured_bin.rows && rect_points[i].x >= 0. && rect_points[i].y >= 0.)
				//????????? ????? ????????? ???????? ???? ?????? 0.7
				if (area > 0.6)
				{
					++cond;
				}
		}
		//????????? ? ?????????? ???????
		--ind;
		//??????????????? ???? ??? ??????? ????????
		if (ind == -1)
			cond = 4;
	}
}

bool check(std::vector<int> vecBit)
{
	if (vecBit.size() == 94)//??????? ?????????????????? ?????? ??????? ?? 94 ????????
		return true;
	else return false;
}

std::vector<int> countStrokes(const cv::Mat& imeg_bin, int intensiv, int location) {
	std::vector<int> count;
	int pixels = 0;
	for (int i = 0; i < imeg_bin.cols; i++) {
		if (int(imeg_bin.at<uchar>(location, i)) == intensiv)
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
	if (intensiv == 255 && int(imeg_bin.at<uchar>(location, 0)) == intensiv)
	{
		count.erase(count.begin());
	}
	return count;
}
std::vector<int> normalizeVec(std::vector<int> histB, std::vector<int> histW) {
	std::vector<int> normalizeVec(histB.size() + histW.size(), 0);
	int zeroMod = histB[0];
	if (histB.size() > 29 && histW.size() > 28) {
		histB[1] = zeroMod;
		histB[14] = zeroMod;
		histB[15] = zeroMod;
		histB[28] = zeroMod;
		histB[29] = zeroMod;
		histW[0] = zeroMod;
		histW[13] = zeroMod;
		histW[14] = zeroMod;
		histW[15] = zeroMod;
		histW[28] = zeroMod;
	}
	
	int j = 0;
	for (size_t i = 0; i < normalizeVec.size()-1; i+=2)
	{
		normalizeVec[i] =round(histB[j] / (double)zeroMod); 
		if (normalizeVec[i] > 4)
		{
			normalizeVec[i] = 4;
		}
		normalizeVec[i + 1] = round(histW[j] / (double)zeroMod);
		if (normalizeVec[i+1] > 4)
		{
			normalizeVec[i+1] = 4;
		}
		j++;
	}
	normalizeVec[normalizeVec.size() - 1] = round(histB[histB.size() - 1] / (double)zeroMod);
	return normalizeVec;
}
std::vector<int> normalizeVecBit(std::vector<int> histB, std::vector<int> histW) {
	std::vector<int> normalizeVec;
	int sizeV = histB.size() + histW.size();
	int zeroMod = histB[0];
	if (histB.size() > 29 && histW.size() > 28) {
		histB[1] = zeroMod;
		histB[14] = zeroMod;
		histB[15] = zeroMod;
		histB[28] = zeroMod;
		histB[29] = zeroMod;
		histW[0] = zeroMod;
		histW[13] = zeroMod;
		histW[14] = zeroMod;
		histW[15] = zeroMod;
		histW[28] = zeroMod;
	}
	int j = 0;
	int indx = 0;
	for (int i = 0; i < histB.size() + histW.size() - 1; i += 2)
	{
		indx = round(histB[j] / (double)zeroMod);
		if (indx > 4)
		{
			indx = 4;
		}
		for (int i = 0; i < indx; i++)
		{
			normalizeVec.push_back(1);
		}
		indx = round(histW[j] / (double)zeroMod);
		if (indx > 4)
		{
			indx = 4;
		}
		for (int i = 0; i < indx; i++)
		{
			normalizeVec.push_back(0);
		}
		j++;
	}
//	normalizeVec[normalizeVec.size() - 1] = round(histB[histB.size() - 1] / (double)zeroMod);
	return normalizeVec;
}
std::vector<int> normalizeVecBit(std::vector<int> vec) {
	std::vector<int> normalizeVec;
	int indx;
	for (int i = 0; i < vec.size() - 1; i += 2)
	{
		indx = vec[i];
		for (int i = 0; i < indx; i++)
		{
			normalizeVec.push_back(1);
		}
		indx = vec[i+1];
		for (int i = 0; i < indx; i++)
		{
			normalizeVec.push_back(0);
		}
	}

	return normalizeVec;
}
cv::Mat drawCode(const cv::Mat& imeg_bin, std::vector<int> normalizeVec)
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
	//print(barcode, "NormalizeBarCode");
	return barcode;
}

std::vector<int> decoder(const std::vector<int>& vecBit)
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
std::string decoderStr(const std::vector<int>& vecBit)
{
	if (check(vecBit) == false)
	{
		return "Failed to recognize the barcode";
	}
	std::string result("");
	std::string firstNum("");
	bool dontFoundFirstNum(false);
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
	std::map<std::string, std::pair<int, std::string>>::iterator it;
	for (int i(3); i < 44; i += 7)
	{
		std::string encod = "";
		for (int j = i; j - i < 7; ++j) {
			encod = encod + std::to_string(vecBit[j]);
		}
		
		it = encodingNumTable.find(encod);
		if (it != encodingNumTable.end())
		{
			std::pair<int, std::string> decod = encodingNumTable.find(encod)->second;
			result += std::to_string(decod.first);
			firstNum += decod.second;
		}
		else
		{
			result +="*";
			dontFoundFirstNum = true;
		}
	}
	for (int i(50); i < 92; i += 7)
	{
		std::string encod = "";
		for (int j = i; j - i < 7; ++j) {
			encod = encod + std::to_string(vecBit[j]);
		}
		it = encodingNumTable.find(encod);
		
		if (it != encodingNumTable.end())
		{
			std::pair<int, std::string> decod = encodingNumTable.find(encod)->second;
			result += std::to_string(decod.first);
			firstNum += decod.second;
		}
		else
		{
			result += "*";
			dontFoundFirstNum = true;
		}
	}
	if (dontFoundFirstNum)
	{
		result = "*" + result;
	}
	else {
		result = std::to_string(encodingFirstNumTable.find(firstNum)->second)+result;
	}
	return result;
}
void findBarcode(const cv::Mat &src, cv::Point2f* rect_points, double scale)
{
	cv::Mat img = src;
	cv::Mat src_gray, src_bin;
	cv::cvtColor(img, src_gray, cv::COLOR_BGR2GRAY);
	
	double sc = 0;
	src_gray = scaler(src_gray, sc, scale);
	
	cv::Mat grad = Gradient(src_gray);
	
	contours(grad, rect_points);
	
	contoursScaler(img, rect_points, sc, scale);
	
}

void approxIm(const cv::Mat &src_bin, cv::Mat &dst)
{
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hireachy;
	//? ??????? ?????????? ??????? ??????? ??? ??????? ?? ???????????
	cv::findContours(src_bin, contours, hireachy, cv::RETR_CCOMP, cv::CHAIN_APPROX_TC89_KCOS, cv::Point());
	double eps = 1;
	cv::Mat approx(src_bin.rows, src_bin.cols, CV_8UC1,cv::Scalar(255));
	for (int i = 0; i < contours.size(); i++)
	{
		cv::approxPolyDP(contours[i], contours[i], eps, true);
		cv::drawContours(approx, contours, i, cv::Scalar(0), -1);
	}
	dst = approx;
}

void findBarcode(const cv::Mat &src, cv::Point2f* rect_points)
{
	cv::Mat img = src;
	cv::Mat src_gray, src_bin;
	cv::cvtColor(img, src_gray, cv::COLOR_BGR2GRAY);

	double sc = 0;
	src_gray = scaler(src_gray, sc, 600.);
	cv::Mat grad = Gradient(src_gray);
	contours(grad, rect_points);
	
	contoursScaler(img, rect_points, sc, 600.);
}
cv::Mat Histogram(std::vector<int> str, std::vector<int>& range) {
	int maxEl = *max_element(str.begin(), str.end());
	
	std::vector<int> histogram(maxEl+2,0);
	for (int i = 0; i < str.size(); i++)
	{
		histogram[str[i]]++;
	}

	int flag = 1;
	for (int i = 1; i < histogram.size()-1; i++)
	{
		if (histogram[i] == 0 && histogram[i + 1]>0)
		{
			range.push_back(i + 1);
		}
		if (histogram[i] > 0 && histogram[i + 1] == 0)
		{
			range.push_back(i);
		}
	}
	
	int hist_w = 512; int hist_h = 512;
	cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(255));

	int max = histogram[0];
	for (int i = 1; i < histogram.size(); i++) {
		if (max < histogram[i]) {
			max = histogram[i];
		}
	}

	for (int i = 0; i < histogram.size(); i++) {
		histogram[i] = ((double)histogram[i] / max) * histImage.rows;
	}

	for (int i = 0; i < histogram.size(); i++)
	{
		rectangle(histImage, cv::Point(i * 2, hist_h - histogram[i]), cv::Point(i * 2 + 2, hist_h ), cv::Scalar(0, 0, 0), -1);
	}

	return histImage;
}
void Normalize_and_read_Barcode(const cv::Mat& src,cv::Mat& dst, cv::Point2f * rect_points, std::vector<int>& vec, std::vector<int>& vecBit)
{
	cv::Scalar color = cv::Scalar(0, 255, 0);
	cv::Point a, b;
	cv::Mat rot, rotImg;
	cv::RotatedRect rotRect;
	cv::Rect roi;
	cv::Mat imgRoi, imgRoiBin;
	std::vector<int> Whist;
	std::vector<int> Bhist;
	cv::Point2f rot_points[4]{ rect_points[0],rect_points[1],rect_points[2],rect_points[3] };
	bool stop = false;
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
	
	
	rot = Rotation(src, a, b, rot_points);
	
	rotRect = cv::RotatedRect(rot_points[0], rot_points[1], rot_points[2]);
	roi = rotRect.boundingRect2f();
	//cv::rectangle(rot, roi, cv::Scalar(0, 255, 0), 5);
	
	
	imgRoi = rot(roi);
	
	cv::cvtColor(imgRoi, imgRoi, cv::COLOR_BGR2GRAY);
	cv::threshold(imgRoi, imgRoi, 100, 255, cv::THRESH_TRIANGLE);
	//print(imgRoi, "ROi_std");
	
	
	Whist = countStrokes(imgRoi, 255, imgRoi.rows / 2);
	Bhist = countStrokes(imgRoi, 0, imgRoi.rows / 2);
	if (Whist.size() != 0 && Bhist.size()!=0)
	{
		Whist = claster(Whist);
		Bhist = claster(Bhist);
	}
		
	vec = normalizeVec(Bhist, Whist);
	vecBit = normalizeVecBit(vec);
	dst= drawCode(imgRoi, vec);
}
void Normalize_and_read_Barcode_with_Blur(const cv::Mat& src, cv::Mat& dst, cv::Point2f* rect_points, std::vector<int>& vec, std::vector<int>& vecBit)
{
	cv::Scalar color = cv::Scalar(0, 255, 0);
	cv::Point a, b;
	cv::Mat rot, rotImg;
	cv::RotatedRect rotRect;
	cv::Rect roi;
	cv::Mat imgRoi, imgRoiBin;
	std::vector<int> Whist;
	std::vector<int> Bhist;
	cv::Point2f rot_points[4]{ rect_points[0],rect_points[1],rect_points[2],rect_points[3] };
	bool stop = false;
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
	rot = Rotation(src, a, b, rot_points);
	rotRect = cv::RotatedRect(rot_points[0], rot_points[1], rot_points[2]);
	roi = rotRect.boundingRect2f();
	imgRoi = rot(roi);
	//imgRoi = sharp(imgRoi);
	cv::medianBlur(imgRoi, imgRoi, 5);
	
	cv::cvtColor(imgRoi, imgRoi, cv::COLOR_BGR2GRAY);
	cv::threshold(imgRoi, imgRoi, 100, 255, cv::THRESH_TRIANGLE);
	//print(imgRoi, "ROi_blur");
	
	Whist = countStrokes(imgRoi, 255, imgRoi.rows / 2);
	Bhist = countStrokes(imgRoi, 0, imgRoi.rows / 2);
	if (Whist.size() != 0 && Bhist.size() != 0)
	{
		Whist = claster(Whist);
		Bhist = claster(Bhist);
	}
	vec = normalizeVec(Bhist, Whist);
	vecBit = normalizeVecBit(vec);
	dst = drawCode(imgRoi, vec);
}
void Normalize_and_read_Barcode_with_APPROX(const cv::Mat& src, cv::Mat& dst, cv::Point2f* rect_points, std::vector<int>& vec, std::vector<int>& vecBit)
{
	cv::Scalar color = cv::Scalar(0, 255, 0);
	cv::Point a, b;
	cv::Mat rot, rotImg;
	cv::RotatedRect rotRect;
	cv::Rect roi;
	cv::Mat imgRoi, imgRoiBin;
	std::vector<int> Whist;
	std::vector<int> Bhist;
	cv::Point2f rot_points[4]{ rect_points[0],rect_points[1],rect_points[2],rect_points[3] };
	bool stop = false;
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
	rot = Rotation(src, a, b, rot_points);
	rotRect = cv::RotatedRect(rot_points[0], rot_points[1], rot_points[2]);
	roi = rotRect.boundingRect2f();
	imgRoi = rot(roi);
	//imgRoi = sharp(imgRoi);
	//cv::medianBlur(imgRoi, imgRoi, 5);

	cv::cvtColor(imgRoi, imgRoi, cv::COLOR_BGR2GRAY);
	cv::threshold(imgRoi, imgRoi, 100, 255,  cv::THRESH_TRIANGLE);
	approxIm(~imgRoi,imgRoi);
	//print(imgRoi, "ROi_approx");
	Whist = countStrokes(imgRoi, 255, imgRoi.rows / 2);
	Bhist = countStrokes(imgRoi, 0, imgRoi.rows / 2);
	if (Whist.size() != 0 && Bhist.size() != 0)
	{
		Whist = claster(Whist);
		Bhist = claster(Bhist);
	}
	vec = normalizeVec(Bhist, Whist);
	
	vecBit = normalizeVecBit(vec);
	
	dst = drawCode(imgRoi, vec);
}
void Normalize_Barcode(const cv::Mat& src, cv::Mat& dst, cv::Point2f* rect_points)
{
	cv::Scalar color = cv::Scalar(0, 255, 0);
	cv::Point a, b;
	cv::Mat rot, rotImg;
	cv::RotatedRect rotRect;
	cv::Rect roi;
	cv::Mat imgRoi, imgRoiBin;
	std::vector<int> Whist;
	std::vector<int> Bhist;
	cv::Point2f rot_points[4]{ rect_points[0],rect_points[1],rect_points[2],rect_points[3] };
	bool stop = false;
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


	rot = Rotation(src, a, b, rot_points);

	rotRect = cv::RotatedRect(rot_points[0], rot_points[1], rot_points[2]);
	roi = rotRect.boundingRect2f();
	//cv::rectangle(rot, roi, cv::Scalar(0, 255, 0), 5);
	//imgRoi = rot(roi);
	dst= rot(roi);
}
void readBarcode(const cv::Mat& src, cv::Mat& dst,  std::vector<int>& vec, std::vector<int>& vecBit) {
	cv::Mat imgRoi = src;
	std::vector<int> Whist;
	std::vector<int> Bhist;
	cv::cvtColor(imgRoi, imgRoi, cv::COLOR_BGR2GRAY);
	cv::threshold(imgRoi, imgRoi, 100, 255, cv::THRESH_OTSU);
	//print(imgRoi, "ROi_std");
	Whist = countStrokes(imgRoi, 255, imgRoi.rows / 2);
	Bhist = countStrokes(imgRoi, 0, imgRoi.rows / 2);
	if (Whist.size() != 0 && Bhist.size() != 0)
	{
		Whist = claster(Whist);
		Bhist = claster(Bhist);
	}
	vec = normalizeVec(Bhist, Whist);
	vecBit = normalizeVecBit(vec);
	dst = drawCode(imgRoi, vec);
}
void readBarcode_withBlur(const cv::Mat& src, cv::Mat& dst,  std::vector<int>& vec, std::vector<int>& vecBit) {
	cv::Mat imgRoi = src;
	std::vector<int> Whist;
	std::vector<int> Bhist;
	cv::cvtColor(imgRoi, imgRoi, cv::COLOR_BGR2GRAY);
	cv::threshold(imgRoi, imgRoi, 100, 255, cv::THRESH_OTSU);
	//print(imgRoi, "ROi_std");
	Whist = countStrokes(imgRoi, 255, imgRoi.rows / 2);
	Bhist = countStrokes(imgRoi, 0, imgRoi.rows / 2);
	if (Whist.size() != 0 && Bhist.size() != 0)
	{
		Whist = claster(Whist);
		Bhist = claster(Bhist);
	}
	vec = normalizeVec(Bhist, Whist);
	vecBit = normalizeVecBit(vec);
	dst = drawCode(imgRoi, vec);
}
void readBarcode_withApprox(const cv::Mat& src, cv::Mat& dst,  std::vector<int>& vec, std::vector<int>& vecBit) {
	cv::Mat imgRoi = src;
	std::vector<int> Whist;
	std::vector<int> Bhist;
	cv::cvtColor(imgRoi, imgRoi, cv::COLOR_BGR2GRAY);
	cv::threshold(imgRoi, imgRoi, 100, 255, cv::THRESH_OTSU);
	//print(imgRoi, "ROi_std");
	Whist = countStrokes(imgRoi, 255, imgRoi.rows / 2);
	Bhist = countStrokes(imgRoi, 0, imgRoi.rows / 2);
	if (Whist.size() != 0 && Bhist.size() != 0)
	{
		Whist = claster(Whist);
		Bhist = claster(Bhist);
	}
	vec = normalizeVec(Bhist, Whist);
	vecBit = normalizeVecBit(vec);
	dst = drawCode(imgRoi, vec);
}
void readBarcode_triangle(const cv::Mat& src, cv::Mat& dst,  std::vector<int>& vec, std::vector<int>& vecBit) {
	cv::Mat imgRoi = src;
	std::vector<int> Whist;
	std::vector<int> Bhist;
	cv::cvtColor(imgRoi, imgRoi, cv::COLOR_BGR2GRAY);
	cv::threshold(imgRoi, imgRoi, 100, 255, cv::THRESH_TRIANGLE);
	//print(imgRoi, "ROi_std");
	Whist = countStrokes(imgRoi, 255, imgRoi.rows / 2);
	Bhist = countStrokes(imgRoi, 0, imgRoi.rows / 2);
	if (Whist.size() != 0 && Bhist.size() != 0)
	{
		Whist = claster(Whist);
		Bhist = claster(Bhist);
	}
	vec = normalizeVec(Bhist, Whist);
	vecBit = normalizeVecBit(vec);
	dst = drawCode(imgRoi, vec);
}
void readBarcode_triangle_withBlur(const cv::Mat& src, cv::Mat& dst,  std::vector<int>& vec, std::vector<int>& vecBit) {
	cv::Mat imgRoi = src;
	std::vector<int> Whist;
	std::vector<int> Bhist;
	cv::cvtColor(imgRoi, imgRoi, cv::COLOR_BGR2GRAY);
	cv::threshold(imgRoi, imgRoi, 100, 255, cv::THRESH_TRIANGLE);
	//print(imgRoi, "ROi_std");
	Whist = countStrokes(imgRoi, 255, imgRoi.rows / 2);
	Bhist = countStrokes(imgRoi, 0, imgRoi.rows / 2);
	if (Whist.size() != 0 && Bhist.size() != 0)
	{
		Whist = claster(Whist);
		Bhist = claster(Bhist);
	}
	vec = normalizeVec(Bhist, Whist);
	vecBit = normalizeVecBit(vec);
	dst = drawCode(imgRoi, vec);
}
void readBarcode_triangle_withApprox(const cv::Mat& src, cv::Mat& dst, std::vector<int>& vec, std::vector<int>& vecBit) {
	cv::Mat imgRoi = src;
	std::vector<int> Whist;
	std::vector<int> Bhist;
	cv::cvtColor(imgRoi, imgRoi, cv::COLOR_BGR2GRAY);
	cv::threshold(imgRoi, imgRoi, 100, 255, cv::THRESH_TRIANGLE);
	//print(imgRoi, "ROi_std");
	Whist = countStrokes(imgRoi, 255, imgRoi.rows / 2);
	Bhist = countStrokes(imgRoi, 0, imgRoi.rows / 2);
	if (Whist.size() != 0 && Bhist.size() != 0)
	{
		Whist = claster(Whist);
		Bhist = claster(Bhist);
	}
	vec = normalizeVec(Bhist, Whist);
	vecBit = normalizeVecBit(vec);
	dst = drawCode(imgRoi, vec);
}
void findnScanBarcode(const cv::Mat& src,cv::Mat& imgRoi, cv::Point2f* rect_points, std::vector<int>& vecBit)
{
	
	std::vector<int> standartScales = {600,400,200};
	std::vector<int> vec;
	bool stop = false;
	int flag = 0;
	while (!stop)
	{
		findBarcode(src, rect_points, standartScales[flag]);
		
		Normalize_and_read_Barcode(src, imgRoi, rect_points, vec, vecBit);
		++flag;
		if ((vec.size() == 59) || (flag == standartScales.size()))
		{
			stop = true;
		}
	}
	if (check(vecBit) == false)
	{
		Normalize_and_read_Barcode_with_Blur(src, imgRoi, rect_points, vec, vecBit);
	}
	if (check(vecBit) == false)
	{
		Normalize_and_read_Barcode_with_APPROX(src, imgRoi, rect_points, vec, vecBit);
	}
}
void findnScanBarcodeX(const cv::Mat& src, cv::Mat& imgRoi, cv::Point2f* rect_points, std::vector<int>& vec, std::vector<int>& vecBit)
{
	cv::Mat tmp;
	std::vector<int> standartScales = { 600,400,200 };
	vec.clear();
	bool stop = false;
	int flag = 0;
	while (!stop)
	{
		findBarcode(src, rect_points, standartScales[flag]);

		Normalize_Barcode(src, imgRoi, rect_points);
		readBarcode(imgRoi, tmp, vec, vecBit);
		++flag;
		if ((vec.size() == 59) || (flag == standartScales.size()))
		{
			stop = true;
		}
	}
	if (vec.size() == 59) {
		if (check(vecBit) == false)
		{
			readBarcode_withBlur(imgRoi, tmp, vec, vecBit);
		}
		if (check(vecBit) == false)
		{
			readBarcode_withApprox(imgRoi, tmp, vec, vecBit);
		}
		if (check(vecBit) == false)
		{
			readBarcode_triangle(imgRoi, tmp, vec, vecBit);
		}
		if (check(vecBit) == false)
		{
			readBarcode_triangle_withBlur(imgRoi, tmp, vec, vecBit);
		}
		if (check(vecBit) == false)
		{
			readBarcode_triangle_withApprox(imgRoi, tmp, vec, vecBit);
		}
		imgRoi = tmp;
	}
}

float distance(float a, float b)
{
	return abs(a - b);
}

std::vector<int> claster(std::vector<int> input)
{
	int newValue = input[0] * 4;
	input.push_back(newValue);
	std::map<int, float> centers;
	std::vector<int> res;
	std::vector<int> clasterName;
	for (int i = 0; i < input.size(); i++)
	{
		centers[i] = input[i];
		clasterName.push_back(i);
		res.push_back(i);
	}

	while (clasterName.size() > 4)
	{
		float min = 100000;
		int claster1 = -1;
		int claster2 = -1;
		for (int i = 0; i < clasterName.size() - 1; i++)
		{
			for (int j = i + 1; j < clasterName.size(); j++)
			{
				if (distance(centers[clasterName[i]], centers[clasterName[j]]) < min)
				{
					min = distance(centers[clasterName[i]], centers[clasterName[j]]);
					claster1 = clasterName[i];
					claster2 = clasterName[j];
				}

			}

		}
		for (int i = 0; i < res.size(); i++)
		{
			if (res[i] == claster1)
			{
				res[i] = claster2;
			}
		}
		std::vector<int>::iterator it;
		it = std::find(clasterName.begin(), clasterName.end(), claster1);
		clasterName.erase(it);
		//centers.erase(centers.find(claster1));
		float mean = 0;
		int diff = 0;

		for (int i = 0; i < res.size(); i++)
		{
			if (res[i] == claster2)
			{
				mean += input[i];
				diff++;
			}
		}
		mean = mean / diff;
		centers[claster2] = mean;


	}
	std::vector<std::pair<int, int>> out;
	for (int i = 0; i < res.size(); i++)
		out.push_back(std::make_pair(input[i], res[i]));

	std::sort(out.begin(), out.end());

	std::vector < std::pair<int, int>> out_or = out;

	out[0].second = 1;
	int tmp(1);
	for (int i = 1; i < out.size(); i++)
	{
		if (out_or[i].second != out_or[i - 1].second)
			tmp += 1;
		out[i].second = tmp;
	}

	std::map<int, int> out_map;
	for (auto i : out) {
		out_map[i.first] = i.second;
	}

	std::vector<int> out_vect;
	for (auto i : input) {
		out_vect.push_back(out_map[i]);
	}
	out_vect.pop_back();

	return out_vect;
}


#include"barcode_detecting.h"

int main()
{
	cv::Mat img = cv::imread("data/test21.jpg", cv::IMREAD_COLOR);
	cv::Point2f rect_points[4];
	cv::Mat contourIm;
	cv::Scalar color = cv::Scalar(0, 255, 0);
	cv::Mat normalizeCode;
	std::vector<int> vecBit;
	findnScanBarcode(img, normalizeCode, rect_points, vecBit);
	cv::copyTo(img, contourIm, img);
	for (int j = 0; j < 4; j++)
	{
		line(contourIm, rect_points[j], rect_points[(j + 1) % 4], color, 3);
	}
	print(contourIm, "contours");
	print(normalizeCode, "NormalizeBarcode");
	std::string res = decoderStr(vecBit);
	std::cout << "Product ID: ";
	for (auto item : res) {
		std::cout << item;
	}
	cv::waitKey(0);
	return 0;
}
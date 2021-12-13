#include"barcode_detecting.h"
#include<poligon.h>
#include<fstream>
template<char openPar, char sep, char closePar>
class WordDelimitedBy : public std::string
{};
std::istream& read(std::istream& istrm, cv::Point2d& A)
{
	float xin(0);
	float yin(0);
	char openPar = '[';
	char closePar = ']';
	char sep = ',';
	istrm >>openPar>>xin>>sep>>yin>>closePar;
	
	if (istrm.good())
	{
		A.x = xin;
		A.y = yin;
	}
	else std::cout << "bad" << std::endl;
	return istrm;
}
//std::istream& operator>>(std::istream& is, WordDelimitedBy<'[', ',', ']'>& output)
//{
//	std::getline(is, output, '[', ',', ']');
//	return is;
//}
int main()
{
	//std::ofstream out;
	//
	//out.open("results/cods.txt",std::ios::out); // окрываем файл для записи
	std::ifstream in,cods;
	in.open("results/res_changed.txt", std::ios::binary);
	cods.open("results/cods.txt", std::ios::binary);
	//out.clear();
	int unsuc = 0;
	int numofTest = 79;
	int firstTest = 0;
	int lastTest = 49;
	//std::cout << "Test  " << "Normalization  " <<"   ID" << std::endl;
	std::cout << "Test  " << "Localization  " << "\tNormalization  " << "   ID" <<"\t\tTestID"<<"\t\tDecodingRes"<< std::endl;
	for (int i = firstTest; i <= lastTest; i++)
	{
	//int i = 4;
		cv::Mat img = cv::imread("data/test"+std::to_string(i)+".jpg", cv::IMREAD_COLOR);
		//print(img, "Test"+ std::to_string(i));
		cv::Point2f rect_points[4];
		cv::Mat contourIm;
		cv::Scalar color = cv::Scalar(0, 255, 0);
		cv::Mat normalizeCode;
		std::vector<int> vecBit;
		std::vector<int> vec;
		std::string resNorm{ "Failed" };
		std::string resDecod{ "Failed" };
		findnScanBarcodeX(img, normalizeCode, rect_points, vec, vecBit);
		if (vec.size() == 59)
		{
			resNorm = "Success";
		}
		cv::copyTo(img, contourIm, img);
		for (int j = 0; j < 4; j++)
		{
			cv::line(contourIm, rect_points[j], rect_points[(j + 1) % 4], color, 3);
		}
		
		//print(normalizeCode, "NormalizeBarcode");
		std::string res = decoderStr(vecBit);
		//std::cout << "Test" + std::to_string(i)+" Normalization result: "+ resNorm + " ID: " << res << std::endl;
		//std::cout << std::to_string(i) << "\t" << resNorm << "\t\t" << res << std::endl;
		if (res == "Failed to recognize the barcode")
		{
			//print(contourIm, "Result test" + std::to_string(i));
			unsuc++;
			res = "Failed       ";
		}
		cv::imwrite("results/test" + std::to_string(i) + ".jpg", contourIm);
		cv::Point2d test[4];
		
		cv::Point2d p[4];
		cv::Point2d area[4];
		/*if (out.is_open())
		{
			out << res << std::endl;
		}*/
		std::string testDecod;
		if (cods.is_open()) {
			cods >> testDecod;
		}


		if (in.is_open())
		{
			for (int i = 0; i < 4; i++)
			{
				read(in, p[i]);
				area[i] = static_cast<cv::Point2d>(rect_points[i]);
				//test.push_back(points[i]);
			}
		}
		//cv::Point2d p1[4]{ cv::Point2d(100,50),cv::Point2d(200,50),cv::Point2d(200,100),cv::Point2d(100,100) };
		/*for (int i = 0; i < 4; i++)
		{
			std::cout << area[i];
		}
		std::cout << std::endl;
		for (int i = 0; i < 4; i++)
		{
			std::cout << p[i];
		}
		std::cout << std::endl;*/
		std::vector<cv::Point> inter= Interseption(area, p);
		
		//cv::RotatedRect rot(rect_points[0],rect_points[1],rect_points[2]);
		//cv::RotatedRect testR(p[0], p[1], p[2]);
		std::vector<cv::Point> contpoints{ rect_points[0],rect_points[1],rect_points[2],rect_points[3] };
		std::vector<cv::Point> testpoints{ p[0], p[1], p[2],p[3] };
		float contArea = cv::contourArea(contpoints);
		float interArea = 0;
		if (inter.size() > 0) {
			interArea = cv::contourArea(inter);
		}
		 
		float testArea = cv::contourArea(testpoints);
		float resArea = interArea / (contArea + testArea - interArea);
		std::string resLoc = "Failed";
		if (resArea > 0.7)
		{
			resLoc = "Success";
		}
		if (res == testDecod)
		{
			resDecod= "Success";
		}
		std::cout << std::to_string(i) << "\t" << resLoc <<"\t\t" << resNorm << "\t\t" << res << "\t"<<testDecod<< "\t" <<resDecod<< std::endl;
	}
	//out.close();
	std::cout << "Num of test: " << lastTest-firstTest+1 << " Suc=" << lastTest - firstTest + 1 - unsuc << " Unsuc=" << unsuc << std::endl;
	std::string line;
	
	
	
	
	cods.close();
	in.close();
	
	
	cv::waitKey(0);
	return 0;
}
#pragma once
#ifndef BARCODE_DETECTING_H
#define BARCODE_DETECTING_H
#include<opencv2/opencv.hpp>
#include<math.h>


/// @brief ������� ��������� ����������� �� �����
/// @param src �����������
/// @param window_name ��� ����
void print(const cv::Mat &src, std::string window_name);
cv::Mat sharp(const cv::Mat &src);
/// @brief ������� ���������� ������� �����������
/// @param src ������� �����������
/// @param scale ������� ������ ����������� ������������ ��������
/// @param modelSize ����� ������ �����������
/// @return ����������� � ����� ��������
cv::Mat scaler(const cv::Mat& src, double& scale, double modelSize);
/// @brief ������� ���������� ������ �������������� ������� ��� �������� �� ����������� � ���������� ���������
/// @param img ����������� � ���������� ��������� �� ������� ����� ��������� ������������� ������
/// @param rect_points ������ ������ �������������� �������
/// @param scale ����� ������� � �������� ����� �������� ������� �������
/// @param modelSize ������ ������������ �����������
void contoursScaler(const cv::Mat &img, cv::Point2f* rect_points, double scale, double modelSize);
/// @brief ������� ��������������� ��� ���������� �������� �� �������
/// @param contours ������� ��������� ����� ��������
/// @return ��������� ����� �������� ��������������� �� ���������� �������
std::vector<std::vector<cv::Point>>  sortContours(std::vector<std::vector<cv::Point>>& contours);
/// @brief ������� ��������� ��������� ����� ����� �������
/// @param a ����� A
/// @param b ����� B
/// @return ���������� ����� ����� �������
double lineLenght(cv::Point2f a, cv::Point2f b);
/// @brief ������� ��������������� ��� �������� �����������
/// @param img ������� �����������
/// @param a ����� � ����� ������������ ������� ���������� �������
/// @param b ����� B ����� ������������ ������� ���������� �������
/// @return ���������� �����������
cv::Mat Rotation(const cv::Mat &img, cv::Point2f a, cv::Point2f b);
/// @brief ������� ��������������� ��� �������� �����������
/// @param img ������� �����������
/// @param a ����� � ����� ������������ ������� ���������� �������
/// @param b ����� B ����� ������������ ������� ���������� �������
/// @param rect_points ������ ������ �������������� �������
/// @return ���������� �����������
cv::Mat Rotation(const cv::Mat &img, cv::Point2f a, cv::Point2f b, cv::Point2f* rect_points);
/// @brief ������� ��������� ������ ������� ����� ���������� ����������� �� dx � ���������� �� dy
/// @param src_gray ������� ����������� � ��������� ������
/// @return ������ ������� ����� ���������� ����������� �� dx � ���������� �� dy
cv::Mat Gradient(const cv::Mat &src_gray);
/// @brief ������� ��� ���������� ������� ��������� �� �����������
/// @param grad ������� ����������� ���������� ������ ������� ����� ���������� ����������� �� dx � ���������� �� dy ��������� �����������
/// @param rect_points ������ ������ �������������� �������
void contours(const cv::Mat& grad, cv::Point2f* rect_points);
/// @brief ������� ��� �������� ������� �������� ������� ������������������ ���������
/// @param vecBit ������� ������������������ �������� ��� ������ ���������
/// @return ��������� �������� (������� ���� ���������)
bool check(std::vector<int> vecBit);
/// @brief ������� ��� ���������� ������� ���� ������������� �����
/// @param imeg_bin �������������� �����������
/// @param intensiv ������������� ������������ ���� ������� (����� ��� ������) ��������� �������� 0 ��� 255
/// @param location ���������� Y ������ �� ������� ���������� ������ ����
/// @return ������ ���������� �������� ������ ������� ����
std::vector<int> countStrokes(const cv::Mat& imeg_bin, int intensiv, int location);
/// @brief ������� ������������� ������ ���������� �������� ������ ������� ����, 
/// ��� ������������� ���������� ���������� �������� ������ � ������� ��������� �������(1,2,3 ���� 4)
/// @param histB ������ ���������� �������� ������ ������� ����
/// @param histW ������ ���������� �������� ������ �������� ����� �������� ����
/// @return ��������������� ������ ���������� �������� ������ ������� � �������� ���������
std::vector<int> normalizeVec(std::vector<int> histB, std::vector<int> histW);
/// @brief ������� ������������� ������ ���������� �������� ������ ������� ����,
/// �� ������������� ���������� �������� ������� ����������� ������� ������������������ �������������� ����������
/// @param histB ������ ���������� �������� ������ ������� ����
/// @param histW ������ ���������� �������� ������ �������� ����� �������� ����
/// @return ������� ���������� ������� ������������������ �������������� � ���������
std::vector<int> normalizeVecBit(std::vector<int> histB, std::vector<int> histW);
/// @brief ������� ��� ��������� ������������������ ���������
/// @param imeg_bin ������� �� ������� ����� �������������� ��������
/// @param normalizeVec ��������������� ������ ���������� �������� ������ ������� � �������� ���������
/// @return ����������� ������������������ ���������
cv::Mat drawCode(const cv::Mat& imeg_bin, std::vector<int> normalizeVec);
/// @brief ������� ��� ������������� ������� ������������������ ��������� � ���������
/// @param vecBit ������� ������������������ ��������� � ���������
/// @return ������������� ������ � ���� ������� 
std::vector<int> decoder(const std::vector<int> &vecBit);
/// @brief ������� ��� ������������� ������� ������������������ ��������� � ���������
/// @param vecBit ������� ������������������ ��������� � ���������
/// @return ������������� ������ � ���� ������
std::string decoderStr(const std::vector<int> &vecBit);
/// @brief ������� ��������������� ��� ������ ��������� �� ����������� � ��������� ���������
/// @param src �������� �����������
/// @param rect_points ������ ������ �������������� �������
/// @param scale ������� � ������� ���������� ����� �������
void findBarcode(const cv::Mat &src, cv::Point2f* rect_points, double scale);
void approxIm(const cv::Mat &src_bin,cv::Mat &dst);
/// @brief ������� ��������������� ��� ������ ��������� �� �����������
/// @param src �������� �����������
/// @param rect_points ������ ������ �������������� �������
void findBarcode(const cv::Mat &src, cv::Point2f* rect_points);
/// @brief ������� ��������������� �������� � ��������� ������� ������������������ �������������� � ���������
/// @param src �������� ����������� �� ������� ���������� ���������� ������������ ���������
/// @param dst ����������� �� ������� ����� ��������� ��������������� ��������
/// @param rect_points ������ ����������� ������� � ������� ��������� ��������
/// @param vec ������ ���������� �������� ������ ������� � �������� ���������
/// @param vecBit ������� ������������������ ��������� � ���������
void Normalize_and_read_Barcode(const cv::Mat& src, cv::Mat& dst, cv::Point2f* rect_points, std::vector<int>& vec, std::vector<int>& vecBit);
/// @brief ������� ��������������� �������� � �������������� ����������� ����� � ��������� ������� ������������������ �������������� � ���������
/// @param src �������� ����������� �� ������� ���������� ���������� ������������ ���������
/// @param dst ����������� �� ������� ����� ��������� ��������������� ��������
/// @param rect_points ������ ����������� ������� � ������� ��������� ��������
/// @param vec ������ ���������� �������� ������ ������� � �������� ���������
/// @param vecBit ������� ������������������ ��������� � ���������
void Normalize_and_read_Barcode_with_Blur(const cv::Mat& src, cv::Mat& dst, cv::Point2f* rect_points, std::vector<int>& vec, std::vector<int>& vecBit);
void Normalize_Barcode(const cv::Mat& src, cv::Mat& dst, cv::Point2f* rect_points);
void readBarcode(const cv::Mat& src, cv::Mat& dst, std::vector<int>& vec, std::vector<int>& vecBit);
void readBarcode_withBlur(const cv::Mat& src, cv::Mat& dst, std::vector<int>& vec, std::vector<int>& vecBit);
void readBarcode_withApprox(const cv::Mat& src, cv::Mat& dst, std::vector<int>& vec, std::vector<int>& vecBit);
void readBarcode_triangle(const cv::Mat& src, cv::Mat& dst, std::vector<int>& vec, std::vector<int>& vecBit);
void readBarcode_triangle_withBlur(const cv::Mat& src, cv::Mat& dst, std::vector<int>& vec, std::vector<int>& vecBit);
void readBarcode_triangle_withApprox(const cv::Mat& src, cv::Mat& dst, std::vector<int>& vec, std::vector<int>& vecBit);
/// @brief ������� ��������������� ��� ��������� ������ ��������� �� ����������� � ���������� ������� ������������������ �������������� ����������
/// @param src �������� ����������� �� ������� ���������� ����������� ����� ���������
/// @param imgRoi ����������� �� ������� ����� ��������� ��������������� ��������
/// @param rect_points ������ � ������� ����� ��������� ��������� ������ �������������� ������������ ������� � ������� ��������� ��������
/// @param vecBit ������� ������������������ �������������� ����������
void findnScanBarcode(const cv::Mat& src, cv::Mat& imgRoi, cv::Point2f* rect_points, std::vector<int>& vecBit);
void findnScanBarcodeX(const cv::Mat& src, cv::Mat& imgRoi, cv::Point2f* rect_points, std::vector<int>& vec, std::vector<int>& vecBit);
std::vector<int> claster(std::vector<int> input);
#endif
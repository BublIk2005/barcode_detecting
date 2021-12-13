/// @file poligon.h
/// @brief Заголовочный файл библиотеки 
/// содержит определения типов данных, структуры, прототипы функций, перечисления, макросы препроцессора.
#pragma once
#ifndef POLIGON_POLIGON_H
#define POLIGON_POLIGON_H
#include<opencv2/opencv.hpp>
#include<vector>
#include<math.h>
/// @struct Color
/// @brief Структура обозначающая определенный цвет.
/// 
/// Используется для того чтобы обозначить наиболее часто используемые цвета в программе.
struct Color
{
    /// @brief cv::Scalar Класс шаблонов для 4-элементного вектора, производного от Vec.
    cv::Scalar color{ 0,0,0 };
    /// @brief Конструктор копии.
    /// @param color cv::Scalar Класс шаблонов для 4-элементного вектора, производного от Vec.
    Color(cv::Scalar &color)
        :color(color) {}
};
/// @brief Голубой цвет.
/// @param cv::Scalar Класс шаблонов для 4-элементного вектора, производного от Vec.
/// @return struct Color
const Color Blue(cv::Scalar(255, 0, 0));
/// @brief Зеленый цвет.
/// @param cv::Scalar Класс шаблонов для 4-элементного вектора, производного от Vec.
/// @return struct Color
const Color Green(cv::Scalar(0, 255, 0));
/// @brief Красный цвет.
/// @param cv::Scalar Класс шаблонов для 4-элементного вектора, производного от Vec.
/// @return struct Color
const Color Red(cv::Scalar(0, 0, 255));
 

///@brief Функция для записи точки.
/// @param ostrm Выходной поток.
/// @return std::ostream Выходной поток.
std::ostream& writeTo(std::ostream& ostrm, const cv::Point2d &A);
/// @brief Функция для чтения точки.
/// @param istrm Входной поток.
/// @return std::istream Входной поток.
std::istream& readFrom(std::istream& istrm, cv::Point2d& A);
/// @brief Метод проверяющий наличие точки в векторе.
/// @param p точка которая ищется в векторе.
/// @param points векотр в котором проиходит поиск.
/// @return bool
bool compP(const cv::Point2d &p, const std::vector<cv::Point2d> &points);
/// @class Edge
/// @brief Класс описывающий сторону полигона.
class Edge {
public:
    /// @brief Точка A граница отрезка.
    cv::Point2d A{ -1,-1 };
    /// @brief Точка B граница отрезка.
    cv::Point2d B{ -1,-1 };
    /// @brief Толщина линии.
    int thickness{ -1 };
    /// @brief Тип линии.
    int linetype{ 8 };
    /// @brief Количество дробных битов в координатах точки.
    int shift{ 0 };
    /// @brief Конструктор по умолчанию.
    Edge() = default;
    /// @brief Деструктор по умолчанию.
    ~Edge() = default;
    /// @brief Конструктор принимающий границы отрезка.
    /// @param A граница отрезка.
    /// @param B граница отрезка.
    Edge(const cv::Point2d &A,const cv::Point2d &B);
    /// @brief Конструктор принимающий границы отрезка.
    /// @param A граница отрезка.
    /// @param B граница отрезка.
    /// @param thickness толщина линии.
    Edge(const cv::Point2d &A,const cv::Point2d &B, int thickness);
    /// @brief Конструктор принимающий границы отрезка.
    /// @param A граница отрезка.
    /// @param B граница отрезка.
    /// @param thickness толщина линии.
    /// @param linetype тип линии.
    Edge(const cv::Point2d &A,const cv::Point2d &B, int thickness, int linetype);
    /// @brief Конструктор принимающий границы отрезка.
    /// @param A граница отрезка.
    /// @param B граница отрезка.
    /// @param thickness толщина линии.
    /// @param linetype тип линии.
    /// @param shift количество дробных битов в координатах точки.
    Edge(const cv::Point2d &A,const cv::Point2d &B, int thickness, int linetype, int shift);
    /// @brief Конструктор копии.
    /// @param A объект class Edge.
    Edge(const Edge& A);
    /// @brief Функция для отрисовки линии.
    /// @param imeg cv::Mat n-мерный класс плотных массивов.
    void DrawLine(cv::Mat &imeg);
    /// @brief Функция для отрисовки линии.
    /// @param imeg cv::Mat n-мерный класс плотных массивов.
    /// @param color cv::Scalar Класс шаблонов для 4-элементного вектора, производного от Vec.
    void DrawLine(cv::Mat &imeg, cv::Scalar color);
    /// @brief Метод для нахождения длины отрезка.
    /// @return Значение длины отрезка.
    double Lenght() const;
};

/// @brief Функция проверяющая существование точки пересечения двух отрезков.
/// @param A Edge Класс описывающий сторону полигона.
/// @param B Edge Класс описывающий сторону полигона.
/// @return bool
bool crossP(const Edge &A, const Edge &B);
/// @brief Функция проверяющая существование отрезка пересечения двух линий.
/// @param A Edge Класс описывающий сторону полигона.
/// @param B Edge Класс описывающий сторону полигона.
/// @return bool
bool crossLn(const Edge &A, const Edge &B);
/// @brief Векторное произведение.
/// @param A Edge Класс описывающий сторону полигона.
/// @param B Edge Класс описывающий сторону полигона.
/// @return int целое значение.
double vecpro(const Edge &A,const Edge &B);
/// @brief Функция находящая точку пересечения двух отрезков.
/// @param A Edge Класс описывающий сторону полигона.
/// @param B Edge Класс описывающий сторону полигона.
/// @return Point2d Класс описывающий сторону полигона.
cv::Point2d CrossPoint(const Edge &A,const Edge &B);
/// @brief Функция находящая отрезок пересечения двух отрезков.
/// @param A Edge Класс описывающий сторону полигона.
/// @param B Edge Класс описывающий сторону полигона.
/// @return Edge Класс описывающий сторону полигона.
Edge CrossLine(const Edge &A, const Edge &B);
/// @brief Функция проверяющая вершины полигона на соответсвие гипотезам.
/// @param points вектор вершин полигона.
void check1(const std::vector<cv::Point2d> &points);
/// @brief Функция проверяющая стороны полигона на соответсвие гипотезам.
/// @param lines вектор сторон полигона.
void check2(const std::vector<Edge> &lines);
/// class Poligon
/// @brief Класс описывающий объект полигон.
class Poligon {
public:
    /// @brief Вектор содержащий множество точек полигона.
    std::vector<cv::Point2d> points{};
    /// @brief Вектор содержащий множество ребер полигона.
    std::vector<Edge> lines{};
    /// @brief Конструктор полигона по умолчанию.
    Poligon() = default;
    /// @brief Деструктор полигона по умолчанию.
    ~Poligon() = default;
    /// @brief Конструктор копии.
    /// @param P экземпляр класса Poligon.
    Poligon(const Poligon& P);
    /// @brief Конструктор принимающий вектор вершин полигона.
    /// @param cpoints Вектор содержащий множество точек cv::Point2d полигона .
    Poligon(const std::vector<cv::Point2d>& cpoints);
    /// @brief Функция для отрисовки вершин полигона.
    /// @param imeg cv::Mat n-мерный класс плотных массивов.
    /// @param color cv::Scalar Класс шаблонов для 4-элементного вектора, производного от Vec.
    void DrawPoints(cv::Mat &imeg, cv::Scalar color);
    /// @brief Функция для отрисовки ребер полигона.
    /// @param imeg cv::Mat n-мерный класс плотных массивов.
    /// @param color cv::Scalar Класс шаблонов для 4-элементного вектора, производного от Vec.
    void DrawLines(cv::Mat &imeg, cv::Scalar color);
    /// @brief Функция для отрисовки полигона.
    /// @param imeg cv::Mat n-мерный класс плотных массивов.
    /// @param color cv::Scalar Класс шаблонов для 4-элементного вектора, производного от Vec.
    void DrawPoligon(cv::Mat &imeg, cv::Scalar color);
    /// @brief Функция для записи полигона.
    /// @param ostrm Выходной поток.
    /// @return std::ostream Выходной поток.
    std::ostream& write(std::ostream& ostrm) const;
    /// @brief Функция для чтения полигона.
    /// @param istrm Входной поток.
    /// @return std::istream Входной поток.
    std::istream& read(std::istream& istrm);
};
/// @brief Функция проверяющая находится ли точка внутри полигона.
/// @param P class Poligon Полигон для которого проходит проверка.
/// @param a cv::Point2d Точка которая проходит проверку.
/// @return bool
bool Interpoint(const Poligon &P, const cv::Point2d &a);
/// @brief Функция для нахождения длины вектора
/// @param l class Edge Класс описывающий сторону полигона.
/// @return double длину вектора
double veclen(const Edge &l);
/// @brief Функция для нахождения косинуса угла между двумя прямыми
/// @param l1 class Edge Класс описывающий сторону полигона.
/// @param l2 class Edge Класс описывающий сторону полигона.
/// @return double косинус угла между векторами.
double angle(const Edge &l1,const Edge &l2);
/// @brief Функция для сортировки точек. 
/// Сортирует точки по возрастанию угла между горизонтальным вектором проведенным из вершины 
/// с наименьшей координатой y и векторами образованными вершиной с наименьшей координатой y и остальными вершинами полигона.
/// @param points Вектор содержащий множество точек полигона.
void sortPoint(std::vector<cv::Point2d>& points);
/// @brief Функция для нахождения полигона пересечения двух полигонов.
/// @param A Полигон.
/// @param B Полигон.
/// @return Полигон пересечения.
Poligon Interseption(const Poligon &A, const Poligon &B);
std::vector<cv::Point> Interseption(const cv::Point2d* p1, const cv::Point2d* p2);
#endif

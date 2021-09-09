project(barcode_detecting)
add_executable(barcode_detecting barcode_detecting.cpp barcode_detecting.h)
target_link_libraries(barcode_detecting ${OpenCV_LIBS})


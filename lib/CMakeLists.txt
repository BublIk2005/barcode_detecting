project(barcode_detecting)
add_library(barcode_detecting barcode_detecting.cpp barcode_detecting.h)
add_library(poligon  poligon.cpp poligon.h)
set_property(TARGET barcode_detecting PROPERTY CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(barcode_detecting poligon ${OpenCV_LIBS})



set_target_properties(${PROJECT_NAME}
        PROPERTIES
        PUBLIC_HEADER barcode.h
        OUTPUT_NAME "barcode"
        CXX_STANDARD 11
        )
target_include_directories(${PROJECT_NAME} PRIVATE .)

install(TARGETS ${PROJECT_NAME}
        CONFIGURATIONS Debug
        barcode_detecting
        DESTINATION dbg/lib
        PUBLIC_HEADER
        DESTINATION dbg/include)
        
        

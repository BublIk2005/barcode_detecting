project(barcode_detecting_test)
add_executable(barcode_detecting_test test.cpp )


target_link_libraries(${PROJECT_NAME} PUBLIC barcode_detecting)

set_target_properties(${PROJECT_NAME}
        PROPERTIES
        OUTPUT_NAME "barcode_detecting_test"
        )

install(TARGETS ${PROJECT_NAME}
        CONFIGURATIONS Release
        DESTINATION rel/bin)

install(TARGETS ${PROJECT_NAME}
        CONFIGURATIONS Debug
        DESTINATION dbg/bin)

add_custom_target(install_${PROJECT_NAME}
        "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --target install
        DEPENDS ${PROJECT_NAME}
        COMMENT "Installing ${PROJECT_NAME}")
        
        
file(GLOB MY_FILES "data/*.jpg") 
file(COPY ${MY_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)
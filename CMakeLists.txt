cmake_minimum_required(VERSION 3.15.0)
project(SpleeterExe VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TENSORFLOW_PATH "${CMAKE_SOURCE_DIR}/third_party/tensorflow-cpu")
set(TENSORFLOW_LIBS ${TENSORFLOW_PATH}/lib/tensorflow.lib)

include_directories(
    ${TENSORFLOW_PATH}/include
    ${CMAKE_SOURCE_DIR}/src/spleeter/include
)

link_directories(${TENSORFLOW_PATH}/lib)

add_executable(SpleeterExe 
    main.cpp
    src/spleeter/SpleeterProcessor.cpp
)

target_link_libraries(SpleeterExe 
    ${TENSORFLOW_LIBS}
)

add_custom_command(TARGET SpleeterExe POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${TENSORFLOW_PATH}/dll
    $<TARGET_FILE_DIR:SpleeterExe>
)

add_custom_command(TARGET SpleeterExe POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/models
    $<TARGET_FILE_DIR:SpleeterExe>/models
)

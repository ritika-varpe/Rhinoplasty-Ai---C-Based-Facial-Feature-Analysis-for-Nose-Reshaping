cmake_minimum_required(VERSION 3.10)
project(Rhinoplasty_ai)

set(CMAKE_CXX_STANDARD 17)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# XGBoost
find_path(XGBOOST_INCLUDE_DIR xgboost/c_api.h PATHS "C:/xgboost/xgboost-master/include")
#find_library(XGBOOST_LIB xgboost PATHS "C:/xgboost/build/Release")
find_library(XGBOOST_LIB xgboost PATHS "C:/xgboost/build/Release")

include_directories(${XGBOOST_INCLUDE_DIR})

add_executable(Rhinoplasty_ai rhinoplasty.cpp)

target_link_libraries(Rhinoplasty_ai ${OpenCV_LIBS} ${XGBOOST_LIB})

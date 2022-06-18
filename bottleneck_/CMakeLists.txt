cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(bottleneck)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CXX_CMAKE_FLAGS} ${TORCH_CXX_FLAGS}" )


option(DOWNLOAD_MNIST "Download the MNIST dataset from the internet" ON)
if (DOWNLOAD_MNIST)
  message(STATUS "Downloading MNIST dataset")
  execute_process(
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/download_mnist.py
      -d ${CMAKE_BINARY_DIR}/data
    ERROR_VARIABLE DOWNLOAD_ERROR)
  if (DOWNLOAD_ERROR)
    message(FATAL_ERROR "Error downloading MNIST dataset: ${DOWNLOAD_ERROR}")
  endif()
endif()


add_executable(bottleneck  bottleneck.cpp)
target_link_libraries(bottleneck "${TORCH_LIBRARIES}")
set_property(TARGET bottleneck PROPERTY CXX_STANDARD 14)





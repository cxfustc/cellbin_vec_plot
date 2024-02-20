cmake_minimum_required (VERSION 2.8)

SET(CMAKE_BUILD_TYPE "Debug")

project (cellbin_vec_plot)
find_package (OpenCV REQUIRED)
add_executable (cellbin_vec_plot 
      main.cpp
      image.cpp
      img.c
      str.c 
			hash.c
			hash_func.c
      utils.c 
			lasso.cpp
			pre_anno.cpp
    )
target_link_libraries (cellbin_vec_plot ${OpenCV_LIBS} tiff z m pthread)

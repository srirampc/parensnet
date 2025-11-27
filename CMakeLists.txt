cmake_minimum_required(VERSION 3.15)

# Project Name
if(NOT PARENSNET_PROJECT_NAME)
  set(PARENSNET_PROJECT_NAME parensnet)
  set(PARENSNET_PROJECT_VERSION 1.0)
endif(NOT PARENSNET_PROJECT_NAME)

project(
    ${PARENSNET_PROJECT_NAME}
    VERSION ${PARENSNET_PROJECT_VERSION}
    LANGUAGES CXX
)

# Options
OPTION(PARALLEL_HDF5 "Whether to use Parallel HDF5" OFF)
# Additional CMake Files
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/ext/pybind11/tools/")
#
set(CMAKE_VERBOSE_MAKEFILE ON)
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(PROJECT_PYMOD _${PROJECT_NAME}_ext)
set(PROJECT_INSTALL_DIR ${PROJECT_NAME})

#
message(STATUS "CMake Variables/Options::")
message(STATUS " -> PROJECT NAME      :  ${PROJECT_NAME}")
message(STATUS " -> PROJECT VERSION   :  ${PROJECT_VERSION}")
message(STATUS " -> PYMODULE NAME     :  ${PROJECT_PYMOD}")
message(STATUS " -> PYMODULE INSTALL  :  ${PROJECT_INSTALL_DIR}")
message(STATUS " -> CMAKE MODULE PATH :  ${CMAKE_MODULE_PATH}")
message(STATUS " -> CMAKE VERBOSE     :  ${CMAKE_VERBOSE_MAKEFILE}")
message(STATUS " -> BUILD             :  ${CMAKE_BUILD_TYPE}")
message(STATUS " -> PARALLEL_HDF5     :  ${PARALLEL_HDF5}")

# directories
include_directories("${PROJECT_SOURCE_DIR}/include/")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/pybind11)

# pybind11 package
#find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
#find_package(pybind11 CONFIG REQUIRED)
# python module
#python_add_library(${PROJECT_PYMOD} MODULE src/mod_parensnet.cpp WITH_SOABI)
#target_link_libraries(${PROJECT_PYMOD} PRIVATE pybind11::headers)
pybind11_add_module(${PROJECT_PYMOD} MODULE
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/mod_parensnet.cpp)

#
add_compile_definitions(PARENSNET_VERSION=${PROJECT_VERSION})
add_compile_definitions(PARENSNET_MODULE_NAME=${PROJECT_PYMOD})

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP :")
  message(STATUS "  -> version     : ${OpenMP_CXX_VERSION}")
  #message(STATUS "  -> headers     : ${OpenMP_CXX_INCLUDE_DIRS}")
  message(STATUS "  -> libraries   : ${OpenMP_CXX_LIBRARIES}")
  include_directories(${OpenMP_CXX_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_PYMOD} PUBLIC OpenMP::OpenMP_CXX)
endif()

# External header-only library: Eigen
add_library(Eigen INTERFACE)
target_include_directories(Eigen INTERFACE
                           ${CMAKE_CURRENT_SOURCE_DIR}/ext/eigen)
target_link_libraries(${PROJECT_PYMOD} PUBLIC Eigen)

# External header-only library: mxx
add_library(mxx INTERFACE)
target_include_directories(Eigen INTERFACE
                           ${CMAKE_CURRENT_SOURCE_DIR}/ext/mxx/include)
target_link_libraries(${PROJECT_PYMOD} PUBLIC Eigen)

# External header-only library: mxx
add_library(prettyp INTERFACE)
target_include_directories(prettyp INTERFACE
                           ${CMAKE_CURRENT_SOURCE_DIR}/ext/mxx/ext/cxx-prettyprint)
target_link_libraries(${PROJECT_PYMOD} PUBLIC prettyp)



# BLAS, LAPACK and LAPACKE
find_package(BLAS)
if(BLAS_FOUND)
  message(STATUS "BLAS :")
  message(STATUS "  -> headers     : ${BLAS_INCLUDE_DIRS}")
  message(STATUS "  -> libraries   : ${BLAS_LIBRARIES}")
  add_compile_definitions(EIGEN_USE_BLAS)
  target_link_libraries(${PROJECT_PYMOD} PUBLIC BLAS::BLAS)
endif()
# LAPACK
find_package(LAPACK)
if(LAPACK_FOUND)
  message(STATUS "LAPACK :")
  message(STATUS "  -> headers     : ${LAPACK_INCLUDE_DIRS}")
  message(STATUS "  -> libraries   : ${LAPACK_LIBRARIES}")
  add_compile_definitions(EIGEN_USE_LAPACK)
  target_link_libraries(${PROJECT_PYMOD} PUBLIC LAPACK::LAPACK)
  # LAPACKE
  find_package(LAPACKE)
  if(LAPACKE_FOUND)
    message(STATUS "LAPACKE :")
    message(STATUS "  -> headers     : ${LAPACKE_INCLUDE_DIRS}")
    message(STATUS "  -> libraries   : ${LAPACKE_LIBRARIES}")
    add_compile_definitions(EIGEN_USE_LAPACKE)
    target_link_libraries(${PROJECT_PYMOD} PUBLIC ${LAPACKE_LIBRARIES})
  endif()
endif()

# MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
  message(STATUS "MPI :")
  message(STATUS "  -> version     : ${MPI_VERSION}")
  message(STATUS "  -> cxx compiler: ${MPI_CXX_COMPILER}")
  message(STATUS "  -> headers     : ${MPI_INCLUDE_PATH}")
  message(STATUS "  -> flags       : ${MPI_CXX_COMPILE_DEFINITIONS}")
  message(STATUS "  -> libs        : ${MPI_LIBRARIES}")
  #include_directories(${MPI_INCLUDE_PATH})
  #add_definitions(${MPI_COMPILE_FLAGS})
  #target_link_libraries(${PROJECT_PYMOD} PUBLIC ${MPI_LIBRARIES})
  target_link_libraries(${PROJECT_PYMOD} PUBLIC MPI::MPI_CXX)
endif (MPI_FOUND)

# HDF5
if (PARALLEL_HDF5)
  set(HDF5_PREFER_PARALLEL true)
endif (PARALLEL_HDF5)
#set(HDF5_USE_STATIC_LIBRARIES TRUE)
find_package(HDF5 COMPONENTS C REQUIRED)
if (HDF5_FOUND)
  if ((NOT HDF5_IS_PARALLEL) AND PARALLEL_HDF5)
    message(WARNING "Can NOT find parallel HDF5, using serial HDF5 instead.")
  endif ((NOT HDF5_IS_PARALLEL) AND PARALLEL_HDF5)
  message(STATUS "HDF5 :")
  message(STATUS "  -> version     : ${HDF5_VERSION}")
  message(STATUS "  -> is parallel : ${HDF5_IS_PARALLEL}")
  message(STATUS "  -> headers     : ${HDF5_C_INCLUDE_DIRS}")
  message(STATUS "  -> definitions : ${HDF5_C_DEFINITIONS}")
  message(STATUS "  -> lib dirs    : ${HDF5_C_LIBRARY_DIRS}")
  message(STATUS "  -> lib names   : ${HDF5_C_LIBRARY_NAMES}")
  message(STATUS "  -> libs        : ${HDF5_C_LIBRARIES}")
  # add_definitions(${HDF5_C_DEFINITIONS})
  # include_directories(${HDF5_C_INCLUDE_DIRS})
  # target_link_directories(${PROJECT_PYMOD} PUBLIC ${HDF5_C_LIBRARY_DIRS})
  # target_link_libraries(${PROJECT_PYMOD} PUBLIC ${HDF5_C_LIBRARIES})
  target_link_libraries(${PROJECT_PYMOD} PUBLIC HDF5::HDF5)
  if (HDF5_IS_PARALLEL)
    add_compile_definitions(USE_PARALLEL_HDF5)
  endif (HDF5_IS_PARALLEL)
endif (HDF5_FOUND)

#
install(TARGETS ${PROJECT_PYMOD} DESTINATION ${PROJECT_INSTALL_DIR})

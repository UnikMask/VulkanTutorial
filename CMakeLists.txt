Cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
project("VulkanTutorial" VERSION "0.1.0" DESCRIPTION "My work on the vulkan tutorial")

set(GLM_ENABLE_CXX_20 "ON")
set(GLFW_USE_WAYLAND "ON")
FetchContent_Declare(
	glm
	GIT_REPOSITORY	https://github.com/g-truc/glm.git
	GIT_TAG  bf71a834948186f4097caa076cd2663c69a10e1e #refs/tags/1.0.0
)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG e2c92645460f680fd272fd2eed591efb2be7dc31
)
FetchContent_MakeAvailable(glm)
FetchContent_MakeAvailable(glfw)

add_library(VulkanTutorial SHARED "./main.cpp" "./main.h")

message(STATUS $ENV{SHADER_COMPILE_SCRIPT})

add_custom_target(compile_shaders ALL
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/compile_shaders.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    COMMENT "Compiling shaders")

add_custom_target ( compile_tags ALL
    COMMAND ctags -R --c++-kinds=+p --fields=+iaS --extra=+q . 
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Compiling project tags")

add_dependencies(${PROJECT_NAME} compile_shaders)
add_dependencies(${PROJECT_NAME} compile_tags)

target_link_libraries(${PROJECT_NAME} 
    PUBLIC glm::glm
    PUBLIC glfw)


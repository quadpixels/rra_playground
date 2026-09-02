cmake_minimum_required(VERSION 3.8)
project(rra_playground LANGUAGES CXX)

ADD_DEFINITIONS(-D_UNICODE)

set(RRA_PATH "$ENV{USERPROFILE}/Downloads/radeon_raytracing_analyzer")

set(CMAKE_CONFIGURATION_TYPES "Release;Debug")

set(CMAKE_CXX_STANDARD 20)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/glfw)

add_executable(MyRRALoader
  main.cpp
  cpu_renderer.cpp
  scene.cpp
  third_party/imgui/imgui.cpp
  third_party/imgui/imgui_draw.cpp
  third_party/imgui/imgui_tables.cpp
  third_party/imgui/imgui_widgets.cpp
  third_party/imgui/backends/imgui_impl_dx12.cpp
  third_party/imgui/backends/imgui_impl_glfw.cpp)

include_directories(AFTER
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/third_party/imgui
  ${CMAKE_SOURCE_DIR}/third_party/imgui/backends
  ${CMAKE_SOURCE_DIR}/third_party/glfw/include
  ${RRA_PATH}/external/third_party
  ${RRA_PATH}/external/third_party/glm
  ${RRA_PATH}/source/backend
  )

# Set debugging CWD
set_target_properties(MyRRALoader PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:MyRRALoader>")
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT            "MyRRALoader")

# Set linked libraries
target_link_libraries(MyRRALoader debug
  ${RRA_PATH}/build/win/vs2022/backend/Debug/Backend-d.lib
  ${RRA_PATH}/build/win/vs2022/external/rdf/rdf/Debug/amdrdf-d.lib
  ${RRA_PATH}/build/win/vs2022/external/rdf/imported/zstd/Debug/zstd-d.lib
  ${RRA_PATH}/build/win/vs2022/external/system_info_utils/source/Debug/system_info-d.lib
  glfw
  dxcompiler.lib
)

target_link_libraries(MyRRALoader optimized
  ${RRA_PATH}/build/win/vs2022/backend/Release/Backend.lib
  ${RRA_PATH}/build/win/vs2022/external/rdf/rdf/Release/amdrdf.lib
  ${RRA_PATH}/build/win/vs2022/external/rdf/imported/zstd/Release/zstd.lib
  ${RRA_PATH}/build/win/vs2022/external/system_info_utils/source/Release/system_info.lib
  glfw
  dxcompiler.lib
)

#install data files, use TARGET_FILE_DIR to handle both configurations
add_custom_command(
    TARGET MyRRALoader POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/dxil.dll"
            "$<TARGET_FILE_DIR:MyRRALoader>"
    COMMAND_EXPAND_LISTS
    COMMENT "Copying dxil.dll to the binary directory"
)

add_custom_command(
    TARGET MyRRALoader POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/dxcompiler.dll"
            "$<TARGET_FILE_DIR:MyRRALoader>"
    COMMAND_EXPAND_LISTS
    COMMENT "Copying dxcompiler.dll to the binary directory"
)
add_custom_command(
    TARGET MyRRALoader POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/3DMarkSolarBay-20241020-003039.rra"
            "$<TARGET_FILE_DIR:MyRRALoader>"
    COMMAND_EXPAND_LISTS
    COMMENT "Copying sample RRA to the binary directory"
)
add_custom_command(
    TARGET MyRRALoader POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/shaders"
            "$<TARGET_FILE_DIR:MyRRALoader>/shaders"
    COMMAND_EXPAND_LISTS
    COMMENT "Copying shaders to the binary directory"
)

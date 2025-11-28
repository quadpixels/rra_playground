cmake_minimum_required(VERSION 3.8)
project(rra_playground LANGUAGES CXX)

ADD_DEFINITIONS(-D_UNICODE)

set(RRA_PATH "$ENV{USERPROFILE}/Downloads/radeon_raytracing_analyzer")

set(CMAKE_CONFIGURATION_TYPES "Release;Debug")

set(CMAKE_CXX_STANDARD 20)

add_executable(MyRRALoader
  main.cpp
  ${CMAKE_SOURCE_DIR}/imgui/backends/imgui_impl_dx12.cpp
  ${CMAKE_SOURCE_DIR}/imgui/backends/imgui_impl_glfw.cpp
  ${CMAKE_SOURCE_DIR}/imgui/imgui.cpp
  ${CMAKE_SOURCE_DIR}/imgui/imgui_demo.cpp
  ${CMAKE_SOURCE_DIR}/imgui/imgui_draw.cpp
  ${CMAKE_SOURCE_DIR}/imgui/imgui_tables.cpp
  ${CMAKE_SOURCE_DIR}/imgui/imgui_widgets.cpp
)

include_directories(AFTER
  ${RRA_PATH}/external/third_party
  ${RRA_PATH}/source/backend
  ${CMAKE_SOURCE_DIR}/imgui
  ${CMAKE_BINARY_DIR}/CompiledShaders
)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/CompiledShaders)

# Call DXC to compile shader into headers
add_custom_command(TARGET MyRRALoader PRE_LINK
  COMMAND dxc.exe /Zi /Od /Vn"g_pPrimaryRay" /Tlib_6_5 /Fh"${CMAKE_BINARY_DIR}/CompiledShaders/primaryray.hlsl.h" /nologo shaders/primaryray.hlsl -Qembed_debug
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_command(TARGET MyRRALoader PRE_LINK
  COMMAND dxc.exe /Zi /Od /Vn"g_pAoRay" /Tlib_6_5 /Fh"${CMAKE_BINARY_DIR}/CompiledShaders/aoray.hlsl.h" /nologo shaders/aoray.hlsl -Qembed_debug
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_command(TARGET MyRRALoader PRE_LINK
  COMMAND dxc.exe /Zi /Od /EPSMain /Vn"g_pFsQuad_PS" /Tps_4_0 /Fh"${CMAKE_BINARY_DIR}/CompiledShaders/fsquad_ps.hlsl.h" /nologo shaders/fsquad.hlsl -Qembed_debug
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_command(TARGET MyRRALoader PRE_LINK
  COMMAND dxc.exe /Zi /Od /EVSMain /Vn"g_pFsQuad_VS" /Tvs_4_0 /Fh"${CMAKE_BINARY_DIR}/CompiledShaders/fsquad_vs.hlsl.h" /nologo shaders/fsquad.hlsl -Qembed_debug
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Set debugging CWD
set_target_properties(MyRRALoader PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:MyRRALoader>")
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT            "MyRRALoader")

# Set linked libraries
target_link_libraries(MyRRALoader debug
  ${RRA_PATH}/build/win/vs2022/backend/Debug/Backend-d.lib
  ${RRA_PATH}/build/win/vs2022/external/rdf/rdf/Debug/amdrdf-d.lib
  ${RRA_PATH}/build/win/vs2022/external/rdf/imported/zstd/Debug/zstd-d.lib
  ${RRA_PATH}/build/win/vs2022/external/system_info_utils/source/Debug/system_info-d.lib
  dxcompiler.lib
)

target_link_libraries(MyRRALoader optimized
  ${RRA_PATH}/build/win/vs2022/backend/Release/Backend.lib
  ${RRA_PATH}/build/win/vs2022/external/rdf/rdf/Release/amdrdf.lib
  ${RRA_PATH}/build/win/vs2022/external/rdf/imported/zstd/Release/zstd.lib
  ${RRA_PATH}/build/win/vs2022/external/system_info_utils/source/Release/system_info.lib
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
            "${CMAKE_SOURCE_DIR}/glfw3.dll"
            "$<TARGET_FILE_DIR:MyRRALoader>"
    COMMAND_EXPAND_LISTS
    COMMENT "Copying glfw3.dll to the binary directory"
)
add_custom_command(
    TARGET MyRRALoader POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/3DMarkSolarBay-20241020-003039.rra"
            "$<TARGET_FILE_DIR:MyRRALoader>"
    COMMAND_EXPAND_LISTS
    COMMENT "Copying glfw3.dll to the binary directory"
)
add_custom_command(
    TARGET MyRRALoader POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/shaders"
            "$<TARGET_FILE_DIR:MyRRALoader>/shaders"
    COMMAND_EXPAND_LISTS
    COMMENT "Copying shaders to the binary directory"
)
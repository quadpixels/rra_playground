project(rra_playground LANGUAGES CXX)

ADD_DEFINITIONS(-D_UNICODE)

set(RRA_PATH "$ENV{USERPROFILE}/Downloads/radeon_raytracing_analyzer")

set(CMAKE_CONFIGURATION_TYPES "Release")  # Release only for now!

set(CMAKE_CXX_STANDARD 20)

add_executable(MyRRALoader
  main.cpp)

include_directories(AFTER
  ${RRA_PATH}/external/third_party
  ${RRA_PATH}/source/backend
  )

target_link_libraries(MyRRALoader
  ${RRA_PATH}/build/win/vs2022/backend/Release/Backend.lib
  ${RRA_PATH}/build/win/vs2022/backend/Release/Backend.lib
  ${RRA_PATH}/build/win/vs2022/external/rdf/rdf/Release/amdrdf.lib
  ${RRA_PATH}/build/win/vs2022/external/rdf/imported/zstd/Release/zstd.lib
  ${RRA_PATH}/build/win/vs2022/external/system_info_utils/source/Release/system_info.lib
  dxcompiler.lib
)

set(O ${PROJECT_BINARY_DIR}/Release)

#install(DIRECTORY shaders  DESTINATION   ${CMAKE_CURRENT_BINARY_DIR}/shaders)
add_custom_command(
    TARGET MyRRALoader POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/dxil.dll"
            "${O}"
    COMMENT "Copying dxil.dll to the binary directory"
)

add_custom_command(
    TARGET MyRRALoader POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/dxcompiler.dll"
            "${O}"
    COMMENT "Copying dxcompiler.dll to the binary directory"
)
add_custom_command(
    TARGET MyRRALoader POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/glfw3.dll"
            "${O}"
    COMMENT "Copying glfw3.dll to the binary directory"
)
add_custom_command(
    TARGET MyRRALoader POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/3DMarkSolarBay-20241020-003039.rra"
            "${O}"
    COMMENT "Copying glfw3.dll to the binary directory"
)
add_custom_command(
    TARGET MyRRALoader POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/shaders"
            "${O}/shaders"
    COMMENT "Copying shaders to the binary directory"
)
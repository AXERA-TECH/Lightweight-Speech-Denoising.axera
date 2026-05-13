cmake_minimum_required(VERSION 3.10)
project(se_cache_only C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(INC_DIR  ${PROJECT_ROOT}/inc)
set(SRC_DIR  ${PROJECT_ROOT}/src)
set(LIB_DIR  ${PROJECT_ROOT}/lib)

set(BUILD_TARGET "x86" CACHE STRING "Build target: x86 | ax525 | ax620l | ax620q | ax630c | ax637 | ax650")
message(STATUS "BUILD_TARGET = ${BUILD_TARGET}")

if(BUILD_AX650)
    set(BUILD_TARGET "ax650")
endif()

if(BUILD_TARGET STREQUAL "x86")
    set(IS_X86 TRUE)
    set(IS_AX_NPU FALSE)
elseif(BUILD_TARGET STREQUAL "ax525")
    set(IS_X86 FALSE)
    set(IS_AX_NPU TRUE)
    set(IS_AX525_FAMILY TRUE)
    set(IS_AX637_FAMILY FALSE)
    set(IS_AX650_FAMILY FALSE)
    set(IS_AX620_FAMILY FALSE)
elseif(BUILD_TARGET STREQUAL "ax637")
    set(IS_X86 FALSE)
    set(IS_AX_NPU TRUE)
    set(IS_AX525_FAMILY FALSE)
    set(IS_AX637_FAMILY TRUE)
    set(IS_AX650_FAMILY FALSE)
    set(IS_AX620_FAMILY FALSE)
elseif(BUILD_TARGET STREQUAL "ax620l")
    set(IS_X86 FALSE)
    set(IS_AX_NPU TRUE)
    set(IS_AX525_FAMILY FALSE)
    set(IS_AX637_FAMILY FALSE)
    set(IS_AX650_FAMILY FALSE)
    set(IS_AX620_FAMILY TRUE)
elseif(BUILD_TARGET STREQUAL "ax650" OR BUILD_TARGET STREQUAL "ax630c")
    set(IS_X86 FALSE)
    set(IS_AX_NPU TRUE)
    set(IS_AX525_FAMILY FALSE)
    set(IS_AX637_FAMILY FALSE)
    set(IS_AX650_FAMILY TRUE)
    set(IS_AX620_FAMILY FALSE)
elseif(BUILD_TARGET STREQUAL "ax620q")
    set(IS_X86 FALSE)
    set(IS_AX_NPU TRUE)
    set(IS_AX525_FAMILY FALSE)
    set(IS_AX637_FAMILY FALSE)
    set(IS_AX650_FAMILY FALSE)
    set(IS_AX620_FAMILY TRUE)
else()
    message(FATAL_ERROR "Unknown BUILD_TARGET: ${BUILD_TARGET}. Use x86 | ax525 | ax620l | ax620q | ax630c | ax637 | ax650")
endif()

# ========== ONNX Runtime (x86 版本) ==========
if(IS_X86)
    set(ORT_SEARCH_PATHS
        ${PROJECT_ROOT}/third_party/onnxruntime
    )
    if(DEFINED ORT_ROOT AND NOT ORT_ROOT STREQUAL "")
        list(INSERT ORT_SEARCH_PATHS 0 ${ORT_ROOT})
    endif()
    set(ORT_INCLUDE_DIR "")
    set(ORT_LIB_DIR "")
    foreach(p ${ORT_SEARCH_PATHS})
        if(EXISTS "${p}/include/onnxruntime_c_api.h")
            set(ORT_INCLUDE_DIR "${p}/include")
            set(ORT_LIB_DIR     "${p}/lib")
            break()
        endif()
    endforeach()
    if(NOT ORT_INCLUDE_DIR)
        message(FATAL_ERROR "onnxruntime_c_api.h not found. "
                "Set ORT_ROOT or copy onnxruntime to c_infer/third_party/onnxruntime/")
    endif()
    message(STATUS "ONNX Runtime include: ${ORT_INCLUDE_DIR}")
    message(STATUS "ONNX Runtime lib:     ${ORT_LIB_DIR}")
endif()

# ========== AX650 BSP SDK ==========
if(BUILD_TARGET STREQUAL "ax650")
    if(NOT DEFINED BSP_SDK_DIR OR BSP_SDK_DIR STREQUAL "")
        set(BSP_SDK_DIR /home/hy/audio_project/third_party/ax650n_bsp_sdk)
    endif()
    if(NOT EXISTS ${BSP_SDK_DIR})
        message(FATAL_ERROR "AX650 BSP SDK not found at ${BSP_SDK_DIR}. Pass -DBSP_SDK_DIR=<path>")
    endif()
    set(BSP_INC_DIR ${BSP_SDK_DIR}/msp/out/include)
    set(BSP_LIB_DIR ${BSP_SDK_DIR}/msp/out/lib)
    message(STATUS "AX650 BSP SDK: ${BSP_SDK_DIR}")
endif()

if(BUILD_TARGET STREQUAL "ax630c")
    if(NOT DEFINED BSP_SDK_DIR OR BSP_SDK_DIR STREQUAL "")
        set(BSP_SDK_DIR /home/hy/audio_project/third_party/ax620e_bsp_sdk)
    endif()
    if(NOT EXISTS ${BSP_SDK_DIR})
        message(FATAL_ERROR "AX630C BSP SDK not found at ${BSP_SDK_DIR}. Pass -DBSP_SDK_DIR=<path>")
    endif()
    set(BSP_INC_DIR ${BSP_SDK_DIR}/msp/out/arm64_glibc/include)
    set(BSP_LIB_DIR ${BSP_SDK_DIR}/msp/out/arm64_glibc/lib)
    message(STATUS "AX630C BSP SDK: ${BSP_SDK_DIR}")
endif()

if(IS_AX620_FAMILY)
    if(NOT DEFINED BSP_SDK_DIR OR BSP_SDK_DIR STREQUAL "")
        set(BSP_SDK_DIR /home/hy/audio_project/third_party/ax620e_bsp_sdk)
    endif()
    if(NOT EXISTS ${BSP_SDK_DIR})
        message(FATAL_ERROR "${BUILD_TARGET} BSP SDK not found at ${BSP_SDK_DIR}. Pass -DBSP_SDK_DIR=<path>")
    endif()
    set(BSP_INC_DIR ${BSP_SDK_DIR}/msp/out/arm_uclibc/include)
    set(BSP_LIB_DIR ${BSP_SDK_DIR}/msp/out/arm_uclibc/lib)
    message(STATUS "${BUILD_TARGET} BSP SDK: ${BSP_SDK_DIR}")
endif()

if(IS_AX525_FAMILY)
    if(NOT DEFINED BSP_SDK_DIR OR BSP_SDK_DIR STREQUAL "")
        set(BSP_SDK_DIR /opt/ax525_bsp_sdk)
    endif()
    if(NOT EXISTS ${BSP_SDK_DIR})
        message(FATAL_ERROR "AX525 BSP SDK not found at ${BSP_SDK_DIR}. Pass -DBSP_SDK_DIR=<path>")
    endif()
    set(BSP_INC_DIR ${BSP_SDK_DIR}/msp/out/arm_uclibc/include)
    set(BSP_LIB_DIR ${BSP_SDK_DIR}/msp/out/arm_uclibc/lib)
    message(STATUS "AX525 BSP SDK: ${BSP_SDK_DIR}")
endif()

if(IS_AX637_FAMILY)
    message(FATAL_ERROR "AX637 BSP SDK is not provided yet. "
            "The axmodel can be used for board-side inference when a matching BSP/runtime is available.")
endif()

include_directories(
    ${INC_DIR}
    ${LIB_DIR}
    ${LIB_DIR}/rnnoise_src
)
if(IS_X86)
    include_directories(${ORT_INCLUDE_DIR})
else()
    include_directories(${BSP_INC_DIR})
endif()

set(KISS_FFT_SOURCES
    ${LIB_DIR}/rnnoise_src/kiss_fft.c
    ${LIB_DIR}/rnnoise_src/rnnoise_tables.c
    ${LIB_DIR}/rnnoise_src/celt_lpc.c
    ${LIB_DIR}/rnnoise_src/pitch.c
)
set(DSP_SOURCES ${LIB_DIR}/tiny_se_v5_dsp.c)

if(IS_X86)
    set(ALGO_SOURCES ${SRC_DIR}/ax_ai_se_denoise.c)
else()
    set(ALGO_SOURCES ${SRC_DIR}/ax_ai_se_denoise_ax.c)
endif()

add_library(se_unified STATIC
    ${KISS_FFT_SOURCES}
    ${DSP_SOURCES}
    ${ALGO_SOURCES}
)
target_include_directories(se_unified PUBLIC
    ${INC_DIR}
    ${LIB_DIR}
    ${LIB_DIR}/rnnoise_src
)
if(IS_X86)
    target_include_directories(se_unified PUBLIC ${ORT_INCLUDE_DIR})
else()
    target_include_directories(se_unified PUBLIC ${BSP_INC_DIR})
endif()

if(IS_X86)
    add_executable(test_cache_denoise
        ${SRC_DIR}/test_cache_denoise.c
    )
    target_include_directories(test_cache_denoise PRIVATE
        ${INC_DIR}
        ${LIB_DIR}
        ${ORT_INCLUDE_DIR}
    )
    target_link_directories(test_cache_denoise PRIVATE
        ${ORT_LIB_DIR}
    )
    target_link_libraries(test_cache_denoise
        se_unified
        onnxruntime
        m
    )
    set_target_properties(test_cache_denoise PROPERTIES
        BUILD_RPATH "${ORT_LIB_DIR}"
        INSTALL_RPATH "${ORT_LIB_DIR}"
    )
else()
    add_executable(test_se_denoise_ax
        ${SRC_DIR}/test_se_denoise_ax.c
    )
    target_include_directories(test_se_denoise_ax PRIVATE
        ${INC_DIR}
        ${LIB_DIR}
        ${BSP_INC_DIR}
    )
    target_link_directories(test_se_denoise_ax PRIVATE
        ${BSP_LIB_DIR}
    )
    target_link_libraries(test_se_denoise_ax
        se_unified
        ax_engine
        ax_interpreter
        ax_sys
        m
    )
endif()

install(TARGETS se_unified ARCHIVE DESTINATION lib)
install(FILES
    ${INC_DIR}/ax_ai_se_denoise.h
    ${INC_DIR}/ax_base_type.h
    ${LIB_DIR}/tiny_se_v5_dsp.h
    DESTINATION include
)
if(IS_X86)
    install(TARGETS test_cache_denoise RUNTIME DESTINATION bin)
else()
    install(TARGETS test_se_denoise_ax RUNTIME DESTINATION bin)
endif()
install(DIRECTORY ${PROJECT_ROOT}/models/
    DESTINATION models
    FILES_MATCHING PATTERN "*.ini" PATTERN "*.onnx" PATTERN "*.axmodel"
)
install(DIRECTORY ${PROJECT_ROOT}/data/
    DESTINATION data
    FILES_MATCHING PATTERN "*.wav"
)

message(STATUS "========================================")
message(STATUS " Cache-only SE Denoise Configuration")
message(STATUS " (gtcrn + tiny_v5 + conv_se)")
message(STATUS "========================================")
message(STATUS "  C Compiler:  ${CMAKE_C_COMPILER}")
message(STATUS "  System:      ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Build Type:  ${CMAKE_BUILD_TYPE}")
message(STATUS "  Target:      ${BUILD_TARGET}")
if(IS_X86)
    message(STATUS "  Backend:     ONNX Runtime (x86)")
    message(STATUS "  ORT:         ${ORT_INCLUDE_DIR}")
    message(STATUS "  Test bin:    test_cache_denoise")
    message(STATUS "  Usage:")
    message(STATUS "    ./test_cache_denoise in.wav out.wav models/teacher_cache_config.ini")
    message(STATUS "    ./test_cache_denoise in.wav out.wav models/ax620_cache_config.ini")
    message(STATUS "    ./test_cache_denoise in.wav out.wav models/ax525_cache_config.ini")
    message(STATUS "    ./test_cache_denoise in.wav out.wav models/gtcrn_stream_config.ini")
else()
    message(STATUS "  Backend:     AX Engine NPU (${BUILD_TARGET})")
    message(STATUS "  BSP SDK:     ${BSP_SDK_DIR}")
    message(STATUS "  Test bin:    test_se_denoise_ax")
    message(STATUS "  Usage:")
    message(STATUS "    ./test_se_denoise_ax in.wav out.wav models/tiny_v5_ax620q_config.ini")
    message(STATUS "    ./test_se_denoise_ax in.wav out.wav models/ax620_cache_ax620q_config.ini")
    message(STATUS "    ./test_se_denoise_ax in.wav out.wav models/gtcrn_stream_ax620q_config.ini")
    message(STATUS "    ./test_se_denoise_ax in.wav out.wav models/tiny_v5_ax650_config.ini")
    message(STATUS "    ./test_se_denoise_ax in.wav out.wav models/ax620_cache_ax650_config.ini")
    message(STATUS "    ./test_se_denoise_ax in.wav out.wav models/gtcrn_stream_ax650_config.ini")
endif()
message(STATUS "========================================")

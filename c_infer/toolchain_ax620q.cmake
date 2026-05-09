set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

# TOOLCHAIN_DIR 由 build_ax620q.sh 通过 -DTOOLCHAIN_DIR 传入

set(TOOLCHAIN_PREFIX ${TOOLCHAIN_DIR}/bin/arm-AX620E-linux-uclibcgnueabihf-)

set(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}g++)
set(CMAKE_AR           ${TOOLCHAIN_PREFIX}ar)
set(CMAKE_RANLIB       ${TOOLCHAIN_PREFIX}ranlib)
set(CMAKE_STRIP        ${TOOLCHAIN_PREFIX}strip)

set(CMAKE_FIND_ROOT_PATH ${TOOLCHAIN_DIR}/arm-AX620E-linux-uclibcgnueabihf/sysroot)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard")


export PATH := $(MAKEFILE_DIR)/downloads/gcc_embedded/bin/:$(PATH)
TARGET_ARCH := cortex-m4
TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-
TARGET_TOOLCHAIN_ROOT := $(TENSORFLOW_ROOT)$(MAKEFILE_DIR)/downloads/gcc_embedded/bin/

$(eval $(call add_third_party_download,$(GCC_EMBEDDED_URL),$(GCC_EMBEDDED_MD5),gcc_embedded,))
$(eval $(call add_third_party_download,$(CMSIS_URL),$(CMSIS_MD5),cmsis,))

# TODO(b/161478030) : change - Wno - vla to - Wvla and remove - Wno-shadow once
# we have a solution for fixing / avoiding being tripped up by these warnings.
PLATFORM_FLAGS = \
  -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK \
  -DTF_LITE_STATIC_MEMORY \
  -DTF_LITE_MCU_DEBUG_LOG \
  -fmessage-length=0 \
  -fno-exceptions \
  -fno-unwind-tables \
  -ffunction-sections \
  -fdata-sections \
  -funsigned-char \
  -MMD \
  -mcpu=cortex-m4 \
  -mthumb \
  -mabi=aapcs \
  -mfloat-abi=hard \
  -mfpu=fpv4-sp-d16 \
  -Wall \
  -Wextra \
  -Wno-shadow \
  -Wno-vla \
  -Wno-strict-aliasing \
  -Wno-type-limits \
  -Wno-unused-parameter \
  -Wno-missing-field-initializers \
  -Wno-write-strings \
  -Wno-sign-compare \
  -Wunused-function \
  -fno-delete-null-pointer-checks \
  -fomit-frame-pointer \
  -g \
  -Os
CXXFLAGS += $(PLATFORM_FLAGS) -std=gnu++11 -fno-rtti -fno-use-cxa-atexit
CCFLAGS += $(PLATFORM_FLAGS)
LDFLAGS += \
  --specs=nosys.specs \
  -Wl,-Map=${TENSORFLOW_ROOT}$(MAKEFILE_DIR)/gen/$(TARGET).map,--cref \
  -Wl,--gc-sections
BUILD_TYPE := micro
MICROLITE_LIBS := \
  -lm
INCLUDES += \
  -isystem$(MAKEFILE_DIR)/downloads/cmsis/CMSIS/Core/Include/

# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
OBJCOPY := $(TARGET_TOOLCHAIN_PREFIX)objcopy


ifeq ($(TARGET), zephyr_nrf)
  $(eval $(call add_third_party_download,$(ZEPHYR_URL),$(ZEPHYR_MD5),zephyr,setup_zephyr))
  export ZEPHYR_SDK_INSTALL_DIR?=/opt/zephyr-sdk
  export ZEPHYR_BASE?=$(realpath $(MAKEFILE_DIR)/downloads/zephyr)

#  INCLUDES +=  -I$(ZEPHYR_SDK_INSTALL_DIR)/modules/nrfx/

endif


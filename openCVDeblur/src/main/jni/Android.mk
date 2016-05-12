LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
OPENCV_CAMERA_MODULES:=on
OPENCV_INSTALL_MODULES:=on
OPENCV_LIB_TYPE:=SHARED
#===================================================================================================
#include D:/Android/OpenCV-android-sdk/sdk/native/jni/OpenCV.mk
#LOCAL_C_INCLUDE:= D:/Android/OpenCV-android-sdk/sdk/native/jni/include
#===================================================================================================
include D:/Android/OpenCV-3.0.0-android-sdk/sdk/native/jni/OpenCV.mk
LOCAL_C_INCLUDE:= D:/Android/OpenCV-3.0.0-android-sdk/sdk/native/jni/include
LOCAL_MODULE    := mixed_sample
LOCAL_SRC_FILES :=  jni_part.cpp \
                    deblur.cpp
LOCAL_LDLIBS +=  -llog -ldl

include $(BUILD_SHARED_LIBRARY)

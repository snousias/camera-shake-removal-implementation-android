#APP_STL := gnustl_static
#APP_CPPFLAGS := -frtti -fexceptions
#APP_ABI := armeabi-v7a
#APP_PLATFORM := android-14
#NDK_TOOLCHAIN_VERSION := 4.8
#APP_STL := stlport_shared  --> does not seem to contain C++11 features
#APP_STL := gnustl_shared
#Enable c++11 extentions in source code
#APP_CPPFLAGS += -std=c++11
APP_STL := gnustl_static
APP_CPPFLAGS := -frtti -fexceptions
APP_ABI := armeabi-v7a
APP_PLATFORM := android-8
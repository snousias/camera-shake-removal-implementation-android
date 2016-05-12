#include "deblur.h"

using namespace std;
using namespace cv;

Mat deblur(Mat image);

extern "C" {


/*
JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial2_Tutorial2Activity_FindFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba);

JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial2_Tutorial2Activity_FindFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba)
{
    Mat& mGr  = *(Mat*)addrGray;
    Mat& mRgb = *(Mat*)addrRgba;
    vector<KeyPoint> v;

    FastFeatureDetector detector(50);
    detector.detect(mGr, v);
    for( unsigned int i = 0; i < v.size(); i++ )
    {
        const KeyPoint& kp = v[i];
        circle(mRgb, Point(kp.pt.x, kp.pt.y), 10, Scalar(255,0,0,255));
    }
}

*/

JNIEXPORT void JNICALL Java_org_opencv_samples_deblur_DeblurActivity_Deblur(JNIEnv*, jobject, jlong addrIn, jlong addrOut);
JNIEXPORT void JNICALL Java_org_opencv_samples_deblur_DeblurActivity_Deblur(JNIEnv*, jobject,jlong addrIn, jlong addrOut)
{
    Mat& mInput  = *(Mat*)addrIn;
    Mat& mOutput = *(Mat*)addrOut;
    mOutput=deblur(mInput);
}

}
#include <jni.h>
#include <ctime>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <limits>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

//==namespace==
using namespace std;
using namespace cv;

class deBlur{
public:
    std::string filename;
    Mat bilateralResult;
    Mat shockFiltered;
    Mat grayScaleImage;
    Mat color, img;
    double sigmaColor;
    double belta;
    double sigmaSpace;
    double sigmaRange;
    double alpha;
    double freqCut;
    int initsize;
    int iterScaleNum;
    int scales;
    int shockFilterIter;
    double shockFilterdt;
    double scalesDenominator, scalesFactor;
    std::vector<int> kernelArray;
    bool isFinal;
    Mat OriginalBlurredScaledMinSingleChannel;
    Mat OriginalBlurredScaledSingleChannel;
    Mat OriginalBlurred;
    Mat OriginalBlurredScaled;
    Mat InitialPSF;
    int CurrentIterationFactor, sizef;
    double riterDenominator, riterFactor;
    Mat deblurredImage;
    Mat PSFResult;
    //-----------------------------------------------------------------
    //Basic functions
    void initialization(void);
    void analyzeKernel(void);
    void readImage(void);
    void readImage(string);
    void readImage(Mat theImage);
    void resizeBlurredToMinimumScale(void);
    void coreDeconv(void);
    void iterativeDeblurring(void);
    //Core deconvolution
    void getImagePSF(Mat &OriginalBlurredScaled, Mat &OriginalBlurredMinimumScaleSingleChannel, Mat &PSF, int &iterScaleNum, bool &isFinal, Mat &OriginalBlurred, double &sigmaRange);
    //Filter--------------------------------------
    Mat shock_filter(Mat IO, int iter, double dt, double h);
    //FFT-----------------------------------------
    Mat fft2(Mat input);
    Mat ifft2(Mat input);
    //PSF to OTF & reverse------------
    Mat psf2otf(Mat inputPSF, Size finalSize);
    Mat otf2psf(Mat inputOTF, Size outSize);
    //Kernel-------------------------------------
    Mat estK(Mat Prediction, Mat OriginalBlurredScaledSingleChannel, Mat PSF, int numberOfIterations);
    Mat getK(Mat PredictionPadded, Mat BlurredPadded, Mat PSF, double belta);
    Mat getKMulSpectrumSupport(Mat PredictionPadded, Mat BlurredPadded, Mat PSF, double belta);
    Mat delta_kernel(int s);
    //Pad---------------------------------------------------
    Mat paddarray(Mat input, Size padding, std::string method, std::string direction);
    Size getPadSize(Mat f);
    //Grayscale-----------------------------------------------------
    Mat r2g(Mat input);
    Mat equalTo(Mat IO);
    //Deconv-------------------------------------------
    Mat deconv(Mat &Blurred, Mat &PSF, double &w0alpha);
    Mat deconv_fnmulComplexMats(Mat Blurred, Mat PSF, double w0alpha);
    Mat deconv_fn(Mat &Blurred, Mat &PSF, double &w0alpha);
    //Tester----------------------------------------------------
    void testExample(void);
};




Mat equalizeIntensity(const Mat& inputImage);
Mat getChannel(Mat input, int theChannel);
Mat conjMat(Mat src);
void displayImage(Mat image, std::string name);
void displayImage(Mat image, std::string name, int channel);
void printMat(Mat input);
void displayKernel(Mat image, std::string name);
void computeDFT(Mat& image, Mat& dest);
void computeIDFT(Mat& complex, Mat& dest);
void deconvolute(Mat& img, Mat& kernel);
Mat divideComplexMats(Mat InputA, Mat InputB);
Mat roundDouble(Mat A, int chNum);
void mulComplexMats(Mat A, Mat B, Mat &Result, int flag, bool what);
void writeMat2File(Mat input, std::string filename);
void divSpectrums(InputArray _srcA, InputArray _srcB, OutputArray _dst, int flags, bool conjB);
void shift(const cv::Mat& src, cv::Mat& dst, cv::Point2f delta, int fill = cv::BORDER_CONSTANT, cv::Scalar value = cv::Scalar(0, 0, 0, 0));
void printMat(Mat theMat, std::string name);
void normaliseHistogram(Mat &src, Mat&dst);
void equalizeHistogramIntensity(Mat &src, Mat&dst);
Mat blinddeblurmap(Mat _inputImage, int _initsize, int _iterScaleNum, double _alpha, double _freqCut, int _belta, double _sigmaSpace, double _sigmaColor, int _shockFilterIter, double _shockFilterdt);
Mat blinddeblurmap(string _filename, int _initsize, int _iterScaleNum, double _alpha, double _freqCut, int _belta, double _sigmaSpace, double _sigmaColor, int _shockFilterIter, double _shockFilterdt);



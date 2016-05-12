#include <ctime>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <conio.h>
#include <vector>
#include <limits>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

//==namespace==
using namespace std;
using namespace cv;

void bilateralFilter(InputArray src, OutputArray dst, double sigmaColor, double sigmaSpace);

void bilateralFilterImpl(Mat1d src, Mat1d dst, double sigmaColor, double sigmaSpace);

double clamp(double min, double max, double  x)
{
    return
            (x < (min)) ? (min) :
            (x < (max)) ? (x) :
            (max);
}

Vec2d trilinear_interpolation(const Mat mat, const double y, const double x, const double z)
{
    const size_t height = mat.size[0];
    const size_t width = mat.size[1];
    const size_t depth = mat.size[2];

    const size_t y_index = clamp(0, height - 1, static_cast<size_t>(y));
    const size_t yy_index = clamp(0, height - 1, y_index + 1);
    const size_t x_index = clamp(0, width - 1, static_cast<size_t>(x));
    const size_t xx_index = clamp(0, width - 1, x_index + 1);
    const size_t z_index = clamp(0, depth - 1, static_cast<size_t>(z));
    const size_t zz_index = clamp(0, depth - 1, z_index + 1);
    const double y_alpha = y - y_index;
    const double x_alpha = x - x_index;
    const double z_alpha = z - z_index;

    return
            (1.0 - y_alpha) * (1.0 - x_alpha) * (1.0 - z_alpha) * mat.at<double>(y_index, x_index, z_index) +
            (1.0 - y_alpha) * x_alpha       * (1.0 - z_alpha) * mat.at<double>(y_index, xx_index, z_index) +
            y_alpha       * (1.0 - x_alpha) * (1.0 - z_alpha) * mat.at<double>(yy_index, x_index, z_index) +
            y_alpha       * x_alpha       * (1.0 - z_alpha) * mat.at<double>(yy_index, xx_index, z_index) +
            (1.0 - y_alpha) * (1.0 - x_alpha) * z_alpha       * mat.at<double>(y_index, x_index, zz_index) +
            (1.0 - y_alpha) * x_alpha       * z_alpha       * mat.at<double>(y_index, xx_index, zz_index) +
            y_alpha       * (1.0 - x_alpha) * z_alpha       * mat.at<double>(yy_index, x_index, zz_index) +
            y_alpha       * x_alpha       * z_alpha       * mat.at<double>(yy_index, xx_index, zz_index);
}

void bilateralFilter(cv::InputArray _src, cv::OutputArray _dst, double sigmaColor, double sigmaSpace)
{
    cv::Mat src = _src.getMat();

    CV_Assert(src.channels() == 1);

    // bilateralFilterImpl runs with double depth, single channel
    if (src.depth() != CV_64FC1) {
        src = cv::Mat(_src.size(), CV_64FC1);
        _src.getMat().convertTo(src, CV_64FC1);
    }

    cv::Mat dst_tmp = cv::Mat(_src.size(), CV_64FC1);
    bilateralFilterImpl(src, dst_tmp, sigmaColor, sigmaSpace);

    _dst.create(dst_tmp.size(), _src.type());
    dst_tmp.convertTo(_dst.getMat(), _src.type());
}
void bilateralFilterImpl(cv::Mat1d src, cv::Mat1d dst, double sigma_color, double sigma_space)
{
    using namespace cv;
    const size_t height = src.rows, width = src.cols;
    const size_t padding_xy = 2, padding_z = 2;
    double src_min, src_max;
    cv::minMaxLoc(src, &src_min, &src_max);

    const size_t small_height = static_cast<size_t>((height - 1) / sigma_space) + 1 + 2 * padding_xy;
    const size_t small_width = static_cast<size_t>((width - 1) / sigma_space) + 1 + 2 * padding_xy;
    const size_t small_depth = static_cast<size_t>((src_max - src_min) / sigma_color) + 1 + 2 * padding_xy;

    int data_size[] = { small_height, small_width, small_depth };
    cv::Mat data(3, data_size, CV_64FC2);
    data.setTo(0);

    // down sample
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            const size_t small_x = static_cast<size_t>(x / sigma_space + 0.5) + padding_xy;
            const size_t small_y = static_cast<size_t>(y / sigma_space + 0.5) + padding_xy;
            const double z = src.at<double>(y, x) - src_min;
            const size_t small_z = static_cast<size_t>(z / sigma_color + 0.5) + padding_z;

            cv::Vec2d v = data.at<cv::Vec2d>(small_y, small_x, small_z);
            v[0] += src.at<double>(y, x);
            v[1] += 1.0;
            data.at<cv::Vec2d>(small_y, small_x, small_z) = v;
        }
    }

    // convolution
    cv::Mat buffer(3, data_size, CV_64FC2);
    buffer.setTo(0);
    int offset[3];
    offset[0] = &(data.at<cv::Vec2d>(1, 0, 0)) - &(data.at<cv::Vec2d>(0, 0, 0));
    offset[1] = &(data.at<cv::Vec2d>(0, 1, 0)) - &(data.at<cv::Vec2d>(0, 0, 0));
    offset[2] = &(data.at<cv::Vec2d>(0, 0, 1)) - &(data.at<cv::Vec2d>(0, 0, 0));

    for (unsigned int dim = 0; dim < 3; ++dim) { // dim = 3 stands for x, y, and depth
        const int off = offset[dim];
        for (int ittr = 0; ittr < 2; ++ittr) {
            cv::swap(data, buffer);

            for (unsigned int y = 1; y < small_height - 1; ++y) {
                for (unsigned int x = 1; x < small_width - 1; ++x) {
                    cv::Vec2d *d_ptr = &(data.at<cv::Vec2d>(y, x, 1));
                    cv::Vec2d *b_ptr = &(buffer.at<cv::Vec2d>(y, x, 1));
                    for (unsigned int z = 1; z < small_depth - 1; ++z, ++d_ptr, ++b_ptr) {
                        cv::Vec2d b_prev = *(b_ptr - off), b_curr = *b_ptr, b_next = *(b_ptr + off);
                        *d_ptr = (b_prev + b_next + 2.0 * b_curr) / 4.0;
                    } // z
                } // x
            } // y
        } // ittr
    } // dim

    // upsample

    for (cv::MatIterator_<cv::Vec2d> d = data.begin<cv::Vec2d>(); d != data.end<cv::Vec2d>(); ++d)
    {
        (*d)[0] /= (*d)[1] != 0 ? (*d)[1] : 1;
    }

    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            double z = src.at<double>(y, x) - src_min;
            double px = double(x) / sigma_space + padding_xy;
            double py = double(y) / sigma_space + padding_xy;
            double pz = double(z) / sigma_color + padding_z;
            dst.at<double>(y, x) = trilinear_interpolation(data, py, px, pz)[0];
        }
    }
}

/*
namespace cv_extend {
void bilateralFilter(cv::InputArray src, cv::OutputArray dst,double sigmaColor, double sigmaSpace);

void bilateralFilterImpl(cv::Mat1d src, cv::Mat1d dst,double sigmaColor, double sigmaSpace);

template<typename T, typename T_, typename T__> inline T clamp(const T_ min, const T__ max, const T x)
{
return
(x < static_cast<T>(min)) ? static_cast<T>(min) :
(x < static_cast<T>(max)) ? static_cast<T>(x) :
static_cast<T>(max);
}

template<typename T>inline
T
trilinear_interpolation(const Mat mat,const double y,const double x,const double z)
{
const size_t height = mat.size[0];
const size_t width = mat.size[1];
const size_t depth = mat.size[2];

const size_t y_index = clamp(0, height - 1, static_cast<size_t>(y));
const size_t yy_index = clamp(0, height - 1, y_index + 1);
const size_t x_index = clamp(0, width - 1, static_cast<size_t>(x));
const size_t xx_index = clamp(0, width - 1, x_index + 1);
const size_t z_index = clamp(0, depth - 1, static_cast<size_t>(z));
const size_t zz_index = clamp(0, depth - 1, z_index + 1);
const double y_alpha = y - y_index;
const double x_alpha = x - x_index;
const double z_alpha = z - z_index;

return
(1.0 - y_alpha) * (1.0 - x_alpha) * (1.0 - z_alpha) * mat.at<T>(y_index, x_index, z_index) +
(1.0 - y_alpha) * x_alpha       * (1.0 - z_alpha) * mat.at<T>(y_index, xx_index, z_index) +
y_alpha       * (1.0 - x_alpha) * (1.0 - z_alpha) * mat.at<T>(yy_index, x_index, z_index) +
y_alpha       * x_alpha       * (1.0 - z_alpha) * mat.at<T>(yy_index, xx_index, z_index) +
(1.0 - y_alpha) * (1.0 - x_alpha) * z_alpha       * mat.at<T>(y_index, x_index, zz_index) +
(1.0 - y_alpha) * x_alpha       * z_alpha       * mat.at<T>(y_index, xx_index, zz_index) +
y_alpha       * (1.0 - x_alpha) * z_alpha       * mat.at<T>(yy_index, x_index, zz_index) +
y_alpha       * x_alpha       * z_alpha       * mat.at<T>(yy_index, xx_index, zz_index);
}
} // end of namespace cv_extend

void cv_extend::bilateralFilter(cv::InputArray _src, cv::OutputArray _dst, double sigmaColor, double sigmaSpace)
{
cv::Mat src = _src.getMat();

CV_Assert(src.channels() == 1);

// bilateralFilterImpl runs with double depth, single channel
if (src.depth() != CV_64FC1) {
src = cv::Mat(_src.size(), CV_64FC1);
_src.getMat().convertTo(src, CV_64FC1);
}

cv::Mat dst_tmp = cv::Mat(_src.size(), CV_64FC1);
bilateralFilterImpl(src, dst_tmp, sigmaColor, sigmaSpace);

_dst.create(dst_tmp.size(), _src.type());
dst_tmp.convertTo(_dst.getMat(), _src.type());
}

void cv_extend::bilateralFilterImpl(cv::Mat1d src, cv::Mat1d dst, double sigma_color, double sigma_space)
{
using namespace cv;
const size_t height = src.rows, width = src.cols;
const size_t padding_xy = 2, padding_z = 2;
double src_min, src_max;
cv::minMaxLoc(src, &src_min, &src_max);

const size_t small_height = static_cast<size_t>((height - 1) / sigma_space) + 1 + 2 * padding_xy;
const size_t small_width = static_cast<size_t>((width - 1) / sigma_space) + 1 + 2 * padding_xy;
const size_t small_depth = static_cast<size_t>((src_max - src_min) / sigma_color) + 1 + 2 * padding_xy;

int data_size[] = { small_height, small_width, small_depth };
cv::Mat data(3, data_size, CV_64FC2);
data.setTo(0);

// down sample
for (unsigned int y = 0; y < height; ++y) {
for (unsigned int x = 0; x < width; ++x) {
const size_t small_x = static_cast<size_t>(x / sigma_space + 0.5) + padding_xy;
const size_t small_y = static_cast<size_t>(y / sigma_space + 0.5) + padding_xy;
const double z = src.at<double>(y, x) - src_min;
const size_t small_z = static_cast<size_t>(z / sigma_color + 0.5) + padding_z;

cv::Vec2d v = data.at<cv::Vec2d>(small_y, small_x, small_z);
v[0] += src.at<double>(y, x);
v[1] += 1.0;
data.at<cv::Vec2d>(small_y, small_x, small_z) = v;
}
}

// convolution
cv::Mat buffer(3, data_size, CV_64FC2);
buffer.setTo(0);
int offset[3];
offset[0] = &(data.at<cv::Vec2d>(1, 0, 0)) - &(data.at<cv::Vec2d>(0, 0, 0));
offset[1] = &(data.at<cv::Vec2d>(0, 1, 0)) - &(data.at<cv::Vec2d>(0, 0, 0));
offset[2] = &(data.at<cv::Vec2d>(0, 0, 1)) - &(data.at<cv::Vec2d>(0, 0, 0));

for (unsigned int dim = 0; dim < 3; ++dim) { // dim = 3 stands for x, y, and depth
const int off = offset[dim];
for (int ittr = 0; ittr < 2; ++ittr) {
cv::swap(data, buffer);

for (unsigned int y = 1; y < small_height - 1; ++y) {
for (unsigned int x = 1; x < small_width - 1; ++x) {
cv::Vec2d *d_ptr = &(data.at<cv::Vec2d>(y, x, 1));
cv::Vec2d *b_ptr = &(buffer.at<cv::Vec2d>(y, x, 1));
for (unsigned int z = 1; z < small_depth - 1; ++z, ++d_ptr, ++b_ptr) {
cv::Vec2d b_prev = *(b_ptr - off), b_curr = *b_ptr, b_next = *(b_ptr + off);
*d_ptr = (b_prev + b_next + 2.0 * b_curr) / 4.0;
} // z
} // x
} // y
} // ittr
} // dim

// upsample

for (cv::MatIterator_<cv::Vec2d> d = data.begin<cv::Vec2d>(); d != data.end<cv::Vec2d>(); ++d)
{
(*d)[0] /= (*d)[1] != 0 ? (*d)[1] : 1;
}

for (unsigned int y = 0; y < height; ++y) {
for (unsigned int x = 0; x < width; ++x) {
const double z = src.at<double>(y, x) - src_min;
const double px = static_cast<double>(x) / sigma_space + padding_xy;
const double py = static_cast<double>(y) / sigma_space + padding_xy;
const double pz = static_cast<double>(z) / sigma_color + padding_z;
dst.at<double>(y, x) = trilinear_interpolation<cv::Vec2d>(data, py, px, pz)[0];
}
}
}

*/

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
    void coreDeconv(Mat OriginalBlurredScaled, Mat OriginalBlurredMinimumScaleSingleChannel, Mat PSF, int iterScaleNum, bool isFinal, Mat OriginalBlurred, double sigmaRange);
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
    Mat deconv(Mat Blurred, Mat PSF, double w0alpha);
    Mat deconv_TVmulComplexMats(Mat Blurred, Mat PSF, double w0alpha);
    Mat deconv_TV(Mat Blurred, Mat PSF, double w0alpha);
    //Tester----------------------------------------------------
    void testExample(void);
};

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

Mat deBlur::getKMulSpectrumSupport(Mat PredictionPadded, Mat BlurredPadded, Mat PSF, double belta){
    //Initialization

    double minValue, maxValue;

    Mat kernelOut;

    Mat k = Mat::zeros(PredictionPadded.size(), CV_64FC2);
    Mat A = Mat::zeros(PredictionPadded.size(), CV_64FC2);
    Mat b = Mat::zeros(PredictionPadded.size(), CV_64FC2);
    Mat d = Mat::zeros(PredictionPadded.size(), CV_64FC2);
    Mat Temp = Mat::zeros(PredictionPadded.size(), CV_64FC2);
    Mat Temp2 = Mat::zeros(PredictionPadded.size(), CV_64FC2);
    Mat beltaMat = Mat::zeros(PredictionPadded.size(), CV_64FC2);
    Mat beltaChannels[2] = { Mat::zeros(PredictionPadded.size(), CV_64F), Mat::zeros(PredictionPadded.size(), CV_64F) };
    Mat PredictionPaddedIn = Mat::zeros(PredictionPadded.size(), CV_64F);
    Mat BlurredPaddedIn = Mat::zeros(PredictionPadded.size(), CV_64F);
    Mat renta2 = Mat::zeros(PredictionPadded.size(), CV_64F);

    std::vector<Mat> Calc;

    //int kernelSize = PSF.rows;
    kernelOut = delta_kernel(PSF.size().height);

    PredictionPaddedIn = PredictionPadded;
    BlurredPaddedIn = BlurredPadded;

    A = fft2(PredictionPaddedIn);
    b = fft2(BlurredPaddedIn);

    for (int i = 0; i < 1; i++){
        k = psf2otf(kernelOut, A.size());
        //std::cout << k << std::endl;

        //d=conj(A).mul(b)-(conj(A).mul(A)+belta).mul(k);=========================================
        mulSpectrums(b, A, Temp, 0, true);
        mulSpectrums(A, A, Temp2, 0, true);
        beltaChannels[0] = Mat(A.size(), CV_64F, Scalar(belta));
        beltaChannels[1] = Mat(A.size(), CV_64F, Scalar(0));
        merge(beltaChannels, 2, beltaMat);
        Temp2 = Temp2 + beltaMat;
        mulSpectrums(Temp2, k, Temp2, 0, false);
        d = Temp - Temp2;

        //renta = (conj(A.*d).*b - conj(A.*d).*A.*k - belta.*conj(d).*k). / (conj(A.*d).*A.*d + belta*conj(d).*d);======
        //Nominator

        Mat Product1 = Mat::zeros(PredictionPadded.size(), CV_64FC2);
        Mat Product2 = Mat::zeros(PredictionPadded.size(), CV_64FC2);
        Mat Product3 = Mat::zeros(PredictionPadded.size(), CV_64FC2);
        Mat Product4 = Mat::zeros(PredictionPadded.size(), CV_64FC2);
        Mat Product5 = Mat::zeros(PredictionPadded.size(), CV_64FC2);
        Mat Product6 = Mat::zeros(PredictionPadded.size(), CV_64FC2);
        Mat Product7 = Mat::zeros(PredictionPadded.size(), CV_64FC2);
        Mat Product8 = Mat::zeros(PredictionPadded.size(), CV_64FC2);
        Mat Product9 = Mat::zeros(PredictionPadded.size(), CV_64FC2);

        //0
        mulSpectrums(A, d, Product1, 0, false);//mulSpectrums
        mulSpectrums(b, Product1, Product2, 0, true);
        //1
        mulSpectrums(A, d, Product1, 0, false);//mulSpectrums
        mulSpectrums(A, Product1, Product3, 0, true);
        //2
        mulSpectrums(k, d, Product5, 0, true);
        Product6 = belta* Product5;
        //Denom . / (conj(A.*d).*A.*d + belta*conj(d).*d);========================================
        //3
        mulSpectrums(A, d, Product1, 0, false);
        mulSpectrums(A, Product1, Product3, 0, true);
        mulSpectrums(Product3, d, Product7, 0, false);
        //4
        mulSpectrums(d, d, Product8, 0, true);
        Product9 = belta*Product8;

        Mat theNom = Product2 - Product3 - Product6;
        Mat theDeNom = Product7 + Product9;
        divSpectrums(theNom, theDeNom, renta2, 0, false);
        mulSpectrums(renta2, d, renta2, 0, false);
        k = k + renta2;
        kernelOut = otf2psf(k, PSF.size());
        kernelOut = abs(kernelOut);
        cv::minMaxLoc(kernelOut, &minValue, &maxValue);

        for (int i = 0; i < kernelOut.rows; i++){
            for (int j = 0; j < kernelOut.cols; j++){
                if (kernelOut.at<double>(i, j) < (maxValue / this->freqCut)){
                    kernelOut.at<double>(i, j) = 0;
                }
            }
        }

        double s = sum(kernelOut)[0];
        kernelOut = (1 / (maxValue))*kernelOut;
        //kernelOut = ((1.0/(s)))*kernelOut;
        //kernelOut = ((1.0 / (s*s)))*kernelOut;							//#!CAUTION#//
        //std::cout << ((1.0 / (s*s*s*s))) << std::endl;
        //kernelOut = 1 * kernelOut;
    }
    //displayImage(kernelOut, "");
    //std::cout << kernelOut << std::endl;
    return kernelOut;
}

void deBlur::initialization(void){
    this->initsize = 21;
    this->iterScaleNum = 10;
    this->alpha = 5;
    this->freqCut = 20.0;
    this->belta = 2;
    this->sigmaSpace = 50;
    this->sigmaColor = 0;
    this->shockFilterIter = 2;
    this->shockFilterdt = 0.5;
    this->filename = "../images/book3.jpg";
    return;
}

Mat getChannel(Mat src, int theChannel){
    int index = 0;
    Mat spl[3];
    split(src, spl);
    if (theChannel == 0 || theChannel == 1 || theChannel == 2){ index = theChannel; }
    return spl[index];
}

Mat deBlur::fft2(Mat input){
    Mat fft = Mat::zeros(input.size(), CV_64FC2);

    if (input.channels() == 1){
        cv::dft(input, fft, DFT_COMPLEX_OUTPUT, input.rows);
    }
    else{
        exit(37);
    }
    return fft; //Matrix with 2 channels
}

Mat deBlur::ifft2(Mat input){
    Mat inputMat = input;
    Mat inverseTransform = Mat::zeros(input.size(), CV_64FC1);

    if (inputMat.channels() == 2){
        cv::dft(inputMat, inverseTransform, DFT_INVERSE | DFT_SCALE | DFT_REAL_OUTPUT, input.rows);
    }
    else{
        exit(37);
    }
    return inverseTransform;
}

Mat roundDouble(Mat A, int chNum){
    Mat B;
    Mat channels[3];
    A.convertTo(A, CV_32F);
    cv::split(A, channels);
    for (int i = 0; i < A.rows; i++){
        for (int j = 0; j < A.cols; j++){
            channels[0].at<double>(i, j) = (double)((int)(channels[0].at<float>(i, j)*10000.0)) / 10000.0;

            if (chNum > 1)
            {
                channels[1].at<double>(i, j) = (double)((int)(channels[1].at<float>(i, j)*10000.0)) / 10000.0;
            }
            if (chNum > 2)
            {
                channels[2].at<double>(i, j) = (double)((int)(channels[2].at<float>(i, j)*10000.0)) / 10000.0;
            }
        }
    }

    cv::merge(channels, chNum, B);

    return B;
}

Mat deBlur::r2g(Mat input){
    Mat output;
    if (input.channels() == 3){
        //cv::cvtColor(input, output, CV_BGR2GRAY);
        vector<Mat> channels(3);
        split(input, channels);
        output = ((double)1 / (double)3)*(channels[0] + channels[1] + channels[2]);
    }
    else
    {
        output = input;
    }
    //output.convertTo(output, CV_64F);
    return output;
}

void deBlur::testExample(void){
    //BILATERAL FILTER
    bilateralFilter(grayScaleImage, bilateralResult, sigmaColor, sigmaSpace);
    namedWindow("Bilateral fliter", 0);// Create a window for display.
    resizeWindow("Bilateral fliter", 500, 500);
    imshow("Bilateral fliter", bilateralResult);
    //SHOCK FILTER
    shockFiltered = shock_filter(bilateralResult, 30, 0.9, 1);
    namedWindow("shock_filtered", 0);// Create a window for display.
    resizeWindow("shock_filtered", 500, 500);
    imshow("shock_filtered", shockFiltered);
    //TESTING==================================
    //PADDING
    Mat padded;
    padded = paddarray(shockFiltered, cv::Size(3, 3), "zero", "both");
    namedWindow("padded", 0);// Create a window for display.
    resizeWindow("padded", 500, 500);
    imshow("padded", padded);
    //========================================
    //SHIFT====================================
    Mat shiftMat = bilateralResult;
    shift(shiftMat, shiftMat, Point(0, 150), BORDER_WRAP);
    namedWindow("shiftMat", 0);// Create a window for display.
    resizeWindow("shiftMat", 500, 500);
    imshow("shiftMat", shiftMat);
    //========================================
    //FFT====================================
    Mat fftResult;
    fftResult = fft2(shiftMat);
    namedWindow("fftResult", 0);// Create a window for display.
    resizeWindow("fftResult", 500, 500);
    imshow("fftResult", getChannel(fftResult, 0));
    //========================================

    //PSF2OTFTEST====================================
    Mat thePSF = Mat::zeros(50, 50, CV_8UC1);
    thePSF = imread("../Images/psf.jpg", 0);
    Size theSize(21, 21);
    resize(thePSF, thePSF, theSize);//resize image
    Mat otf = Mat::zeros(150, 150, CV_32FC2);

    vector<Mat> complexOTF;
    split(otf, complexOTF);
    namedWindow("shiftResult2", 0);// Create a window for display.
    resizeWindow("shiftResult2", 500, 500);
    imshow("shiftResult2", complexOTF[0]);
    namedWindow("thePSF", 0);// Create a window for display.
    resizeWindow("thePSF", 500, 500);
    imshow("thePSF", thePSF);
}

cv::Size deBlur::getPadSize(Mat f){
    return Size(((f.rows - 1) / 2), ((f.cols - 1) / 2));
}

cv::Mat deBlur::paddarray(Mat input, Size padding, std::string method, std::string direction){
    Mat padded;
    int top, bottom, left, right;
    int borderType = BORDER_CONSTANT;
    Scalar value;
    top = padding.height;
    bottom = padding.height;
    left = padding.width;
    right = padding.width;
    value = Scalar(0, 0, 0);
    if (direction == "both")
    {
        top = padding.height;
        bottom = padding.height;
        left = padding.width;
        right = padding.width;
    }
    else if (direction == "post")
    {
        top = 0;
        bottom = padding.height;
        left = 0;
        right = padding.width;
    }
    else if (direction == "pre")
    {
        top = padding.height;
        bottom = 0;
        left = padding.width;
        right = 0;
    }
    if (method == "zero")
    {
        borderType = BORDER_CONSTANT;
    }
    else if (method == "replicate")
    {
        borderType = BORDER_REPLICATE;
    }
    copyMakeBorder(input, padded, top, bottom, left, right, borderType, value);
    return padded;
}

Mat divideComplexMats(Mat A, Mat B){
    Mat result, AC1, AC2, BC1, BC2, RC1, RC2, RDENOM, RC1NOM, RC2NOM;
    AC1 = Mat::zeros(A.size(), CV_64FC1);
    AC2 = Mat::zeros(A.size(), CV_64FC1);
    BC1 = Mat::zeros(B.size(), CV_64FC1);
    BC2 = Mat::zeros(B.size(), CV_64FC1);
    RC1 = Mat::zeros(B.size(), CV_64FC1);
    RC2 = Mat::zeros(B.size(), CV_64FC1);
    RDENOM = Mat::zeros(B.size(), CV_64FC1);
    RC1NOM = Mat::zeros(B.size(), CV_64FC1);
    RC2NOM = Mat::zeros(B.size(), CV_64FC1);
    result = Mat::zeros(B.size(), CV_64FC2);

    AC1 = getChannel(A, 0);
    AC2 = getChannel(A, 1);
    BC1 = getChannel(B, 0);
    BC2 = getChannel(B, 1);

    Mat BC1MUL, BC2MUL, AC1BC1MUL, AC2BC2MUL, AC1BC2MUL, AC2BC1MUL;

    BC1MUL = BC1.mul(BC1);
    BC2MUL = BC2.mul(BC2);
    AC1BC1MUL = AC1.mul(BC1);
    AC2BC2MUL = AC2.mul(BC2);
    AC1BC2MUL = AC1.mul(BC2);
    AC2BC1MUL = AC2.mul(BC1);

    RDENOM = BC1MUL + BC2MUL;
    RC1NOM = AC1BC1MUL + AC2BC2MUL;
    RC2NOM = AC2BC1MUL - AC1BC2MUL;

    RC1 = RC1NOM.mul(1 / RDENOM);
    RC2 = RC2NOM.mul(1 / RDENOM);

    //divide(RC1NOM, RDENOM, RC1);
    //divide(RC2NOM, RDENOM, RC2);

    vector<Mat> complexConjugate;
    complexConjugate.push_back(RC1);
    complexConjugate.push_back(RC2);
    //merge(complexConjugate, 2, result);
    merge(complexConjugate, result);
    return result;
}

void mulComplexMats(Mat A, Mat B, Mat &result, int flag, bool what){
    Mat  AC1, AC2, BC1, BC2, RC1, RC2;
    AC1 = Mat::zeros(A.size(), CV_64FC1);
    AC2 = Mat::zeros(A.size(), CV_64FC1);
    BC1 = Mat::zeros(B.size(), CV_64FC1);
    BC2 = Mat::zeros(B.size(), CV_64FC1);
    RC1 = Mat::zeros(B.size(), CV_64FC1);
    RC2 = Mat::zeros(B.size(), CV_64FC1);
    AC1 = getChannel(A, 0);
    AC2 = getChannel(A, 1);
    BC1 = getChannel(B, 0);
    BC2 = getChannel(B, 1);

    RC1 = AC1.mul(BC1) - AC2.mul(BC2);
    RC2 = AC1.mul(BC2) + AC2.mul(BC1);
    vector<Mat> complexConjugate;
    complexConjugate.push_back(RC1);
    complexConjugate.push_back(RC2);
    merge(complexConjugate, result);

    return;
}

void divSpectrums(InputArray _srcA, InputArray _srcB, OutputArray _dst, int flags, bool conjB)
{
    Mat srcA = _srcA.getMat(), srcB = _srcB.getMat();
    int depth = srcA.depth(), cn = srcA.channels(), type = srcA.type();
    int rows = srcA.rows, cols = srcA.cols;
    int j, k;

    CV_Assert(type == srcB.type() && srcA.size() == srcB.size());
    CV_Assert(type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2);

    _dst.create(srcA.rows, srcA.cols, type);
    Mat dst = _dst.getMat();

    bool is_1d = (flags & DFT_ROWS) || (rows == 1 || (cols == 1 &&
                                                      srcA.isContinuous() && srcB.isContinuous() && dst.isContinuous()));

    if (is_1d && !(flags & DFT_ROWS))
        cols = cols + rows - 1, rows = 1;

    int ncols = cols*cn;
    int j0 = cn == 1;
    int j1 = ncols - (cols % 2 == 0 && cn == 1);

    if (depth == CV_32F)
    {
        const float* dataA = srcA.ptr<float>();
        const float* dataB = srcB.ptr<float>();
        float* dataC = dst.ptr<float>();
        float eps = FLT_EPSILON; // prevent div0 problems

        size_t stepA = srcA.step / sizeof(dataA[0]);
        size_t stepB = srcB.step / sizeof(dataB[0]);
        size_t stepC = dst.step / sizeof(dataC[0]);

        if (!is_1d && cn == 1)
        {
            for (k = 0; k < (cols % 2 ? 1 : 2); k++)
            {
                if (k == 1)
                    dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if (rows % 2 == 0)
                    dataC[(rows - 1)*stepC] = dataA[(rows - 1)*stepA] / (dataB[(rows - 1)*stepB] + eps);
                if (!conjB)
                    for (j = 1; j <= rows - 2; j += 2)
                    {
                        double denom = (double)dataB[j*stepB] * dataB[j*stepB] +
                                       (double)dataB[(j + 1)*stepB] * dataB[(j + 1)*stepB] + (double)eps;

                        double re = (double)dataA[j*stepA] * dataB[j*stepB] +
                                    (double)dataA[(j + 1)*stepA] * dataB[(j + 1)*stepB];

                        double im = (double)dataA[(j + 1)*stepA] * dataB[j*stepB] -
                                    (double)dataA[j*stepA] * dataB[(j + 1)*stepB];

                        dataC[j*stepC] = (float)(re / denom);
                        dataC[(j + 1)*stepC] = (float)(im / denom);
                    }
                else
                    for (j = 1; j <= rows - 2; j += 2)
                    {
                        double denom = (double)dataB[j*stepB] * dataB[j*stepB] +
                                       (double)dataB[(j + 1)*stepB] * dataB[(j + 1)*stepB] + (double)eps;

                        double re = (double)dataA[j*stepA] * dataB[j*stepB] -
                                    (double)dataA[(j + 1)*stepA] * dataB[(j + 1)*stepB];

                        double im = (double)dataA[(j + 1)*stepA] * dataB[j*stepB] +
                                    (double)dataA[j*stepA] * dataB[(j + 1)*stepB];

                        dataC[j*stepC] = (float)(re / denom);
                        dataC[(j + 1)*stepC] = (float)(im / denom);
                    }
                if (k == 1)
                    dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
            }
        }

        for (; rows--; dataA += stepA, dataB += stepB, dataC += stepC)
        {
            if (is_1d && cn == 1)
            {
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if (cols % 2 == 0)
                    dataC[j1] = dataA[j1] / (dataB[j1] + eps);
            }

            if (!conjB)
                for (j = j0; j < j1; j += 2)
                {
                    double denom = (double)(dataB[j] * dataB[j] + dataB[j + 1] * dataB[j + 1] + eps);
                    double re = (double)(dataA[j] * dataB[j] + dataA[j + 1] * dataB[j + 1]);
                    double im = (double)(dataA[j + 1] * dataB[j] - dataA[j] * dataB[j + 1]);
                    dataC[j] = (float)(re / denom);
                    dataC[j + 1] = (float)(im / denom);
                }
            else
                for (j = j0; j < j1; j += 2)
                {
                    double denom = (double)(dataB[j] * dataB[j] + dataB[j + 1] * dataB[j + 1] + eps);
                    double re = (double)(dataA[j] * dataB[j] - dataA[j + 1] * dataB[j + 1]);
                    double im = (double)(dataA[j + 1] * dataB[j] + dataA[j] * dataB[j + 1]);
                    dataC[j] = (float)(re / denom);
                    dataC[j + 1] = (float)(im / denom);
                }
        }
    }
    else
    {
        const double* dataA = srcA.ptr<double>();
        const double* dataB = srcB.ptr<double>();
        double* dataC = dst.ptr<double>();
        double eps = DBL_EPSILON; // prevent div0 problems

        size_t stepA = srcA.step / sizeof(dataA[0]);
        size_t stepB = srcB.step / sizeof(dataB[0]);
        size_t stepC = dst.step / sizeof(dataC[0]);

        if (!is_1d && cn == 1)
        {
            for (k = 0; k < (cols % 2 ? 1 : 2); k++)
            {
                if (k == 1)
                    dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if (rows % 2 == 0)
                    dataC[(rows - 1)*stepC] = dataA[(rows - 1)*stepA] / (dataB[(rows - 1)*stepB] + eps);
                if (!conjB)
                    for (j = 1; j <= rows - 2; j += 2)
                    {
                        double denom = dataB[j*stepB] * dataB[j*stepB] +
                                       dataB[(j + 1)*stepB] * dataB[(j + 1)*stepB] + eps;

                        double re = dataA[j*stepA] * dataB[j*stepB] +
                                    dataA[(j + 1)*stepA] * dataB[(j + 1)*stepB];

                        double im = dataA[(j + 1)*stepA] * dataB[j*stepB] -
                                    dataA[j*stepA] * dataB[(j + 1)*stepB];

                        dataC[j*stepC] = re / denom;
                        dataC[(j + 1)*stepC] = im / denom;
                    }
                else
                    for (j = 1; j <= rows - 2; j += 2)
                    {
                        double denom = dataB[j*stepB] * dataB[j*stepB] +
                                       dataB[(j + 1)*stepB] * dataB[(j + 1)*stepB] + eps;

                        double re = dataA[j*stepA] * dataB[j*stepB] -
                                    dataA[(j + 1)*stepA] * dataB[(j + 1)*stepB];

                        double im = dataA[(j + 1)*stepA] * dataB[j*stepB] +
                                    dataA[j*stepA] * dataB[(j + 1)*stepB];

                        dataC[j*stepC] = re / denom;
                        dataC[(j + 1)*stepC] = im / denom;
                    }
                if (k == 1)
                    dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
            }
        }

        for (; rows--; dataA += stepA, dataB += stepB, dataC += stepC)
        {
            if (is_1d && cn == 1)
            {
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if (cols % 2 == 0)
                    dataC[j1] = dataA[j1] / (dataB[j1] + eps);
            }

            if (!conjB)
                for (j = j0; j < j1; j += 2)
                {
                    double denom = dataB[j] * dataB[j] + dataB[j + 1] * dataB[j + 1] + eps;
                    double re = dataA[j] * dataB[j] + dataA[j + 1] * dataB[j + 1];
                    double im = dataA[j + 1] * dataB[j] - dataA[j] * dataB[j + 1];
                    dataC[j] = re / denom;
                    dataC[j + 1] = im / denom;
                }
            else
                for (j = j0; j < j1; j += 2)
                {
                    double denom = dataB[j] * dataB[j] + dataB[j + 1] * dataB[j + 1] + eps;
                    double re = dataA[j] * dataB[j] - dataA[j + 1] * dataB[j + 1];
                    double im = dataA[j + 1] * dataB[j] + dataA[j] * dataB[j + 1];
                    dataC[j] = re / denom;
                    dataC[j + 1] = im / denom;
                }
        }
    }
}



Mat deBlur::psf2otf(Mat inputPSF, Size finalSize){
    Mat otf;
    Mat psf;

    Size padSize = finalSize - inputPSF.size();
    psf = paddarray(inputPSF, padSize, "zero", "post");
    int pixelsShiftX = (int)(floor(inputPSF.size().width / 2));
    int pixelsShiftY = (int)(floor(inputPSF.size().height / 2));

    //std::cout << pixelsShiftY << std::endl;
    shift(psf, psf, Point(-pixelsShiftX, -pixelsShiftY), BORDER_WRAP);
    //displayImage(psf, "");
    otf = fft2(psf);

    return otf;
}

Mat deBlur::otf2psf(Mat OTF, Size outSize){
    //Init
    Mat psf;
    Mat psfResult;
    Mat newPSF;
    vector<Mat> complexOTF;
    if (OTF.channels() == 2){
        psf = ifft2(OTF);
        if (psf.channels() == 1){
            int pixelsShiftX = (int)(floor((outSize.width) / 2));
            int pixelsShiftY = (int)(floor((outSize.height) / 2));
            shift(psf, psf, Point(pixelsShiftX, pixelsShiftY), BORDER_WRAP);
            psfResult = psf(Rect(0, 0, outSize.width, outSize.height));
        }
    }
    else{
        exit(5);
    }
    return psfResult;
}

void deBlur::readImage(){
    Mat B;
    B = this->img;
    B.convertTo(B, CV_64F);
    B = (1.0 / 255.0)*B;
    //B = roundDouble(B,3);
    //std::cout << B << std::endl;
    this->OriginalBlurred = B;
    return;
}

void deBlur::readImage(std::string thefile){
    Mat B;
    this->img = imread(thefile, CV_64F); //loads color if it is available
    B = this->img;
    B.convertTo(B, CV_64F);
    B = (1.0 / 255.0)*B;
    //B = roundDouble(B,3);
    //std::cout << B << std::endl;
    this->OriginalBlurred = B;
    return;
}

void deBlur::readImage(Mat thefile){
    Mat B;
    B = thefile;
    B.convertTo(B, CV_64F);
    B = (1.0 / 255.0)*B;
    //B = roundDouble(B,3);
    //std::cout << B << std::endl;
    this->OriginalBlurred = B;
    return;
}


Mat deBlur::shock_filter(Mat IO, int iter, double dt, double h){
    Mat I_mx, I_px, I_my, I_py, I_x, I_y, Dx, Dy, I_mx_abs, I_px_abs, I_my_abs, I_py_abs, I_xx, I_yy, I_xy;
    Mat I_xy_m_dy, I_xy_p_dy, I_xy_my, I_xy_py;
    Mat a_grad_I, I_nn, I_ee, I_buf;
    Mat I_buf2, I_buf3;
    Mat a2_grad_I, result;
    Mat I_t, I_in;
    Mat kernelx, kernely;
    Mat IShiftColsForward;
    Mat IShiftColsBackwards;
    Mat IShiftRowsForward;
    Mat IShiftRowsBackwards;
    Mat IxShiftRowsForward, IxShiftRowsBackwards;

    int nx = IO.rows;
    int ny = IO.cols;
    double h_2 = pow(h, 2);
    I_in = IO;
    for (int i = 0; i < iter; i++){
        I_x = Mat::zeros(I_in.size(), CV_64F);
        I_px = Mat::zeros(I_in.size(), CV_64F);
        I_mx = Mat::zeros(I_in.size(), CV_64F);
        I_mx_abs = Mat::zeros(I_in.size(), CV_64F);
        I_px_abs = Mat::zeros(I_in.size(), CV_64F);
        I_y = Mat::zeros(I_in.size(), CV_64F);
        I_py = Mat::zeros(I_in.size(), CV_64F);
        I_my = Mat::zeros(I_in.size(), CV_64F);
        I_my_abs = Mat::zeros(I_in.size(), CV_64F);
        I_py_abs = Mat::zeros(I_in.size(), CV_64F);
        I_xx = Mat::zeros(I_in.size(), CV_64F);
        I_yy = Mat::zeros(I_in.size(), CV_64F);
        I_xy = Mat::zeros(I_in.size(), CV_64F);
        Dx = Mat::zeros(I_in.size(), CV_64F);
        Dy = Mat::zeros(I_in.size(), CV_64F);

        IShiftColsForward = paddarray(I_in, cv::Size(0, 1), "replicate", "both");
        IShiftColsForward = IShiftColsForward(Rect(0, 0, I_in.size().width, IO.size().height));

        IShiftColsBackwards = paddarray(I_in, cv::Size(0, 1), "replicate", "both");
        IShiftColsBackwards = IShiftColsBackwards(Rect(0, 2, I_in.size().width, IO.size().height));

        IShiftRowsForward = paddarray(I_in, cv::Size(1, 0), "replicate", "both");
        IShiftRowsForward = IShiftRowsForward(Rect(0, 0, I_in.size().width, IO.size().height));

        IShiftRowsBackwards = paddarray(I_in, cv::Size(1, 0), "replicate", "both");
        IShiftRowsBackwards = IShiftRowsBackwards(Rect(2, 0, I_in.size().width, IO.size().height));

        //	IShiftColsForward = I(:, [1 1:nx - 1]);
        //	IShiftColsBackwards = I(:, [2:nx nx]);
        //	IShiftRowsForward = I([1 1:ny - 1], :);
        //	IShiftRowsBackwards = I([2:ny ny], :);

        I_mx = I_in - IShiftColsForward;
        I_px = IShiftColsBackwards - I_in;

        I_my = I_in - IShiftRowsForward;
        I_py = IShiftRowsBackwards - I_in;
        I_x = (I_mx + I_px) / 2;
        I_y = (I_my + I_py) / 2;
        I_mx_abs = abs(I_mx);
        I_px_abs = abs(I_px);

        I_my_abs = abs(I_my);
        I_py_abs = abs(I_py);
        min(I_mx_abs, I_px_abs, Dx);
        min(I_my_abs, I_py_abs, Dy);

        for (int i = 0; i < Dx.rows; i++){
            for (int j = 0; j < Dx.cols; j++){
                if (I_mx.at<double>(i, j)*I_px.at<double>(i, j) < 0){
                    Dx.at<double>(i, j) = 0;
                }
            }
        }
        for (int i = 0; i < Dy.rows; i++){
            for (int j = 0; j < Dy.cols; j++){
                if (I_my.at<double>(i, j)*I_py.at<double>(i, j) < 0){
                    Dy.at<double>(i, j) = 0;
                }
            }
        }

        I_xx = IShiftColsBackwards + IShiftColsForward - 2 * I_in;
        I_yy = IShiftRowsBackwards + IShiftRowsForward - 2 * I_in;

        IxShiftRowsForward = paddarray(I_x, cv::Size(1, 0), "replicate", "both");
        IxShiftRowsForward = IxShiftRowsForward(Rect(0, 0, I_x.size().width, I_x.size().height));
        IxShiftRowsBackwards = paddarray(I_x, cv::Size(1, 0), "replicate", "both");
        IxShiftRowsBackwards = IxShiftRowsBackwards(Rect(2, 0, I_x.size().width, I_x.size().height));
        I_xy = IxShiftRowsBackwards - IxShiftRowsForward;
        I_xy = I_xy / 2;

        // Compute flow
        a_grad_I = Dx.mul(Dx) + Dy.mul(Dy);
        a_grad_I.convertTo(a_grad_I, CV_64F);
        sqrt(a_grad_I, a_grad_I);

        a2_grad_I = Mat::zeros(I_in.size(), CV_64F);
        I_buf = abs(I_x);
        I_buf2 = abs(I_y);
        a2_grad_I = I_buf + I_buf2;

        double dl = 0.00000001;  // small delta

        //I_nn = I_xx.*abs(I_x). ^ 2 + 2 * I_xy.*I_x.*I_y + I_yy.*abs(I_y). ^ 2;
        //I_nn = I_nn. / (abs(I_x). ^ 2 + abs(I_y). ^ 2 + dl);
        //I_ee = I_xx.*abs(I_y). ^ 2 - 2 * I_xy.*I_x.*I_y + I_yy.*abs(I_x). ^ 2;
        //I_ee = I_ee. / (abs(I_x). ^ 2 + abs(I_y). ^ 2 + dl);
        Mat absI_x, absI_y, absI_xsquare, absI_ysquare, mulI_xyI_xI_y, denom;
        //Compute I_nn & I_ee
        //Create separate additions
        absI_x = abs(I_x);
        absI_y = abs(I_y);
        absI_xsquare = absI_x.mul(absI_x);
        absI_ysquare = absI_y.mul(absI_y);
        mulI_xyI_xI_y = I_xy.mul(I_x);
        mulI_xyI_xI_y = mulI_xyI_xI_y.mul(I_y);
        denom = absI_xsquare + absI_ysquare + dl;
        I_nn = I_xx.mul(absI_xsquare) + 2 * mulI_xyI_xI_y + I_yy.mul(absI_ysquare);
        divide(I_nn, denom, I_nn);
        I_ee = I_xx.mul(absI_xsquare) - 2 * mulI_xyI_xI_y + I_yy.mul(absI_ysquare);
        divide(I_ee, denom, I_ee);

        for (int i = 0; i < a2_grad_I.rows; i++){
            for (int j = 0; j < a2_grad_I.cols; j++){
                if (a2_grad_I.at<double>(i, j) == 0){
                    I_nn.at<double>(i, j) = I_xx.at<double>(i, j);
                    I_ee.at<double>(i, j) = I_yy.at<double>(i, j);
                }
            }
        }

        I_t = Mat::zeros(IO.size(), CV_64FC1);
        //I_t = a2_grad_I / h;
        //displayImage(I_t, "");
        int thesign;
        for (int i = 0; i < I_t.rows; i++){
            for (int j = 0; j < I_t.cols; j++){
                if (I_nn.at<double>(i, j) > 0){
                    thesign = 1;
                }
                else{
                    thesign = -1;
                }
                I_t.at<double>(i, j) = -thesign*a2_grad_I.at<double>(i, j) / h;
            }
        }

        I_in = I_in + dt *I_t;
    }
    //displayImage(I_in, "");
    return I_in;
}

void deBlur::iterativeDeblurring(void){
    int currentIterationFactor = 0;
    double denominator;
    double resizeFactor;
    Mat InitialPSF;
    Mat temp;
    int InitialPSFSize;
    this->isFinal = false;
    //Start iterations

    for (int thisIter = 0; thisIter < this->scales; thisIter++){
        //Check of is final-----------------------------------------------
        if (thisIter == this->scales - 1){
            this->isFinal = true;
        }
        //std::cout << this->isFinal << std::endl;
        //Resize----------------------------------------------------------
        currentIterationFactor = scales - thisIter;// +1;
        denominator = pow(2, currentIterationFactor - 1);
        resizeFactor = 1 / denominator;

        Size newSize((int)(this->OriginalBlurred.rows * resizeFactor), (int)(this->OriginalBlurred.rows * resizeFactor));
        resize(this->OriginalBlurred, this->OriginalBlurredScaled, newSize);

        //Get size--------------------------------------------------------------------------
        InitialPSFSize = this->kernelArray.at(currentIterationFactor - 1); // ctor);
        InitialPSF = Mat::zeros(InitialPSFSize, InitialPSFSize, CV_64F);

        //[deblurredImage, PSFResult] ------------------------------------------------
        coreDeconv(this->OriginalBlurredScaled, this->OriginalBlurredScaledMinSingleChannel, InitialPSF, iterScaleNum, isFinal, this->OriginalBlurred, this->sigmaRange);

        //Bicubic interp-------------------------------------------------------------------
        if (thisIter < scales){
            //OriginalBlurredScaledMinSingleChannel = mycubic_2();
            cv::resize(this->deblurredImage, this->OriginalBlurredScaledMinSingleChannel, this->deblurredImage.size() + this->deblurredImage.size(), 0, 0, CV_INTER_CUBIC);

            //cv::resize(this->OriginalBlurredScaledSingleChannel, this->OriginalBlurredScaledMinSingleChannel, this->deblurredImage.size() + this->deblurredImage.size(), 0, 0, CV_INTER_CUBIC);

            //displayImage(this->OriginalBlurredScaledMinSingleChannel, "");
        }
    }

    return;
}

void shift(const cv::Mat& src, cv::Mat& dst, cv::Point2f delta, int fill, cv::Scalar value) {
    // error checking
    assert(fabs(delta.x) < src.cols && fabs(delta.y) < src.rows);

    // split the shift into integer and subpixel components
    cv::Point2i deltai((int)ceil(delta.x), (int)ceil(delta.y));
    cv::Point2f deltasub(fabs(delta.x - deltai.x), fabs(delta.y - deltai.y));

    // INTEGER SHIFT
    // first create a border around the parts of the Mat that will be exposed
    int t = 0, b = 0, l = 0, r = 0;
    if (deltai.x > 0) l = deltai.x;
    if (deltai.x < 0) r = -deltai.x;
    if (deltai.y > 0) t = deltai.y;
    if (deltai.y < 0) b = -deltai.y;
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, t, b, l, r, fill, value);

    // SUBPIXEL SHIFT
    float eps = std::numeric_limits<float>::epsilon();
    if (deltasub.x > eps || deltasub.y > eps) {
        switch (src.depth()) {
            case CV_32F:
            {
                cv::Matx<float, 1, 2> dx(1 - deltasub.x, deltasub.x);
                cv::Matx<float, 2, 1> dy(1 - deltasub.y, deltasub.y);
                sepFilter2D(padded, padded, -1, dx, dy, cv::Point(0, 0), 0, cv::BORDER_CONSTANT);
                break;
            }
            case CV_64F:
            {
                cv::Matx<double, 1, 2> dx(1 - deltasub.x, deltasub.x);
                cv::Matx<double, 2, 1> dy(1 - deltasub.y, deltasub.y);
                sepFilter2D(padded, padded, -1, dx, dy, cv::Point(0, 0), 0, cv::BORDER_CONSTANT);
                break;
            }
            default:
            {
                cv::Matx<float, 1, 2> dx(1 - deltasub.x, deltasub.x);
                cv::Matx<float, 2, 1> dy(1 - deltasub.y, deltasub.y);
                padded.convertTo(padded, CV_32F);
                sepFilter2D(padded, padded, CV_32F, dx, dy, cv::Point(0, 0), 0, cv::BORDER_CONSTANT);
                break;
            }
        }
    }

    // construct the region of interest around the new matrix
    cv::Rect roi = cv::Rect(std::max(-deltai.x, 0), std::max(-deltai.y, 0), 0, 0) + src.size();
    dst = padded(roi);
}

void deBlur::resizeBlurredToMinimumScale(void){
    this->scales = this->kernelArray.size();
    this->scalesDenominator = pow((scales - 1), 2);
    this->scalesFactor = 1 / scalesDenominator;
    cv::Size s = this->OriginalBlurred.size();
    cv::Size s2 = Size((int)((double)s.height * scalesFactor), (int)((double)s.width * scalesFactor));
    resize(this->OriginalBlurred, this->OriginalBlurredScaledMinSingleChannel, s2);   //resize image
    OriginalBlurredScaledMinSingleChannel = r2g(OriginalBlurredScaledMinSingleChannel);
    return;
}

Mat deBlur::deconv_TV(Mat Blurred, Mat PSF, double w0alpha){
    Mat L, delta;
    Mat delta0, delta1, delta2, delta3, delta4, delta5, delta6, delta12, delta345;
    Mat Fk0, Fkx, Fky, Fkxx, Fkyy, Fkxy, FFTK, FFTy, FFTL, FFTLnom, FFTLdenom;
    Mat FFTLdenom1, FFTLdenom2, FFTLdenom3;
    Mat k0 = (Mat_<double>(1, 1) << 1);
    Mat kx = (Mat_<double>(1, 2) << 1, -1);
    Mat ky = (Mat_<double>(2, 1) << 1, -1);
    Mat kxx = (Mat_<double>(1, 3) << 1, -2, 1);
    Mat kyy = (Mat_<double>(3, 1) << 1, -2, 1);
    Mat kxy = (Mat_<double>(2, 2) << 1, -1, -1, 1);

    Fk0 = psf2otf(k0, Blurred.size());
    Fkx = psf2otf(kx, Blurred.size());
    Fky = psf2otf(ky, Blurred.size());
    Fkxx = psf2otf(kxx, Blurred.size());
    Fkyy = psf2otf(kyy, Blurred.size());
    Fkxy = psf2otf(kxy, Blurred.size());

    //delta = w0alpha*(conj(Fk0).mul(Fk0)) + 0.5*((conj(Fkx).mul(Fkx)) + (conj(Fky).mul(Fky))) + 0.25*((conj(Fkxx).mul(Fkxx)) + (conj(Fkxy).mul(Fkxy)) + (conj(Fkyy).mul(Fkyy)));
    Mat temp;
    mulSpectrums(Fk0, Fk0, delta0, 0, true);//mulSpectrums
    mulSpectrums(Fkx, Fkx, delta1, 0, true);
    mulSpectrums(Fky, Fky, delta2, 0, true);;
    mulSpectrums(Fkxx, Fkxx, delta3, 0, true);
    mulSpectrums(Fkxy, Fkxy, delta4, 0, true);
    mulSpectrums(Fkyy, Fkyy, delta5, 0, true);

    delta12 = delta1 + delta2;
    delta345 = delta3 + delta4 + delta5;
    delta = w0alpha*(delta0 + 0.5*delta12 + 0.25*delta345);

    Mat in = Blurred;
    //FFts Blurred Image & PSF---------------------------------------------------------------------------------------
    FFTy = fft2(in);
    FFTK = psf2otf(PSF, in.size());
    //====Test psf-=============================================
    //Mat test = (Mat_<double>(5, 5) << 0.014504096341232, 0, 0, 0, 0, 0, 0.046744586253541, 0.112844238837864, 0.051426510893680, 0, 0.017762321574491, 0.121914237762706, 0.251036943066299, 0.119463962537351, 0.018568266253068, 0, 0.053770085514989, 0.115725926085497, 0.040091529611319, 0, 0, 0, 0, 0.013680666173926, 0.022466629094036);
    //FFTK = psf2otf(test, Blurred.size());
    //FFTK = psf2otf(delta_kernel(5), Blurred.size());
    //======================================================
    //%FFTLnom = (FFTy.*conj(FFTK)).*delta;
    //%FFTLdenom = ((FFTK.*conj(FFTK)).*delta + (Fkx.*conj(Fkx)) + (Fky.*conj(Fky)));
    //%FFTL = FFTLnom. / FFTLdenom;
    Mat Product1, Product2;
    mulSpectrums(FFTy, FFTK, Product1, 0, true);
    mulSpectrums(Product1, delta, Product2, 0, false);
    Mat Product3, Product4, Product5;
    mulSpectrums(FFTK, FFTK, Product3, 0, true);
    mulSpectrums(Product3, delta, Product4, 0, false);
    FFTLnom = Product2;
    FFTLdenom = Product4 + delta1 + delta2;
    divSpectrums(FFTLnom, FFTLdenom, FFTL, 0, false);
    //FFTL = divideComplexMats(FFTLnom, FFTLdenom);
    //ifft
    L = ifft2(FFTL);

    //displayImage(L, "");
    return L;
}

Mat deBlur::deconv_TVmulComplexMats(Mat Blurred, Mat PSF, double w0alpha){
    Mat L, delta;
    Mat delta0, delta1, delta2, delta3, delta4, delta5, delta6, delta12, delta345;
    Mat Fk0, Fkx, Fky, Fkxx, Fkyy, Fkxy, FFTK, FFTy, FFTL, FFTLnom, FFTLdenom;
    Mat FFTLdenom1, FFTLdenom2, FFTLdenom3;
    Mat k0 = (Mat_<double>(1, 1) << 1);
    Mat kx = (Mat_<double>(1, 2) << 1, -1);
    Mat ky = (Mat_<double>(2, 1) << 1, -1);
    Mat kxx = (Mat_<double>(1, 3) << 1, -2, 1);
    Mat kyy = (Mat_<double>(3, 1) << 1, -2, 1);
    Mat kxy = (Mat_<double>(2, 2) << 1, -1, -1, 1);
    Fk0 = psf2otf(k0, Blurred.size());
    Fkx = psf2otf(kx, Blurred.size());
    Fky = psf2otf(ky, Blurred.size());
    Fkxx = psf2otf(kxx, Blurred.size());
    Fkyy = psf2otf(kyy, Blurred.size());
    Fkxy = psf2otf(kxy, Blurred.size());
    //delta = w0alpha*(conj(Fk0).mul(Fk0)) + 0.5*((conj(Fkx).mul(Fkx)) + (conj(Fky).mul(Fky))) + 0.25*((conj(Fkxx).mul(Fkxx)) + (conj(Fkxy).mul(Fkxy)) + (conj(Fkyy).mul(Fkyy)));
    Mat temp;
    temp = conjMat(Fk0);
    mulComplexMats(Fk0, temp, delta0, 0, false);//mulSpectrums
    temp = conjMat(Fkx);
    mulComplexMats(Fkx, temp, delta1, 0, false);
    temp = conjMat(Fky);
    mulComplexMats(Fky, temp, delta2, 0, false);
    temp = conjMat(Fkxx);
    mulComplexMats(Fkxx, temp, delta3, 0, false);
    temp = conjMat(Fkxy);
    mulComplexMats(Fkxy, temp, delta4, 0, false);
    temp = conjMat(Fkyy);
    mulComplexMats(Fkyy, temp, delta5, 0, false);
    delta12 = delta1 + delta2;
    delta345 = delta3 + delta4 + delta5;
    delta = w0alpha*(delta0 + 0.5*delta12 + 0.25*delta345);
    //FFts Blurred Image & PSF---------------------------------------------------------------------------------------
    FFTy = fft2(Blurred);
    FFTK = psf2otf(PSF, Blurred.size());
    //====Test psf-=============================================
    //Mat test = (Mat_<double>(5, 5) << 0.014504096341232, 0, 0, 0, 0, 0, 0.046744586253541, 0.112844238837864, 0.051426510893680, 0, 0.017762321574491, 0.121914237762706, 0.251036943066299, 0.119463962537351, 0.018568266253068, 0, 0.053770085514989, 0.115725926085497, 0.040091529611319, 0, 0, 0, 0, 0.013680666173926, 0.022466629094036);
    //FFTK = psf2otf(test, Blurred.size());
    //FFTK = psf2otf(delta_kernel(5), Blurred.size());
    //======================================================
    //%FFTLnom = (FFTy.*conj(FFTK)).*delta;
    //%FFTLdenom = ((FFTK.*conj(FFTK)).*delta + (Fkx.*conj(Fkx)) + (Fky.*conj(Fky)));
    //%FFTL = FFTLnom. / FFTLdenom;
    Mat Product1, Product2;
    temp = conjMat(FFTK);
    mulComplexMats(FFTy, temp, Product1, 0, false);
    mulComplexMats(Product1, delta, Product2, 0, false);
    Mat Product3, Product4, Product5;
    temp = conjMat(FFTK);
    mulComplexMats(FFTK, temp, Product3, 0, false);
    mulComplexMats(Product3, delta, Product4, 0, false);
    FFTLnom = Product2;
    FFTLdenom = Product4 + delta1 + delta2;
    FFTL = divideComplexMats(FFTLnom, FFTLdenom);
    //ifft
    L = ifft2(FFTL);

    return L;
}

Mat deBlur::deconv(Mat Blurred, Mat PSF, double w0alpha){
    Mat theResult, deblurred_image;
    Size ks;
    ks = getPadSize(PSF);

    int channelsNumber = Blurred.channels();
    vector<Mat> allTheChannels;
    vector<Mat> deconved;
    cv::split(Blurred, allTheChannels);
    for (int j = 0; j < channelsNumber; j++){
        Mat in = allTheChannels.at(j);  //Blurred;
        in = paddarray(in, ks, "replicate", "both");		//#!CAUTION!#//
        theResult = deconv_TV(in, PSF, w0alpha);
        Size d = theResult.size();
        // Setup a rectangle to define your region of interest
        int xmin = ks.height;
        int ymin = ks.width;
        int height = (d.height - ks.height - ks.height);
        int width = (d.width - ks.width - ks.width);
        cv::Rect myROI(xmin, ymin, width, height);

        deblurred_image = theResult(myROI);
        deconved.push_back(deblurred_image);
        normalize(deblurred_image, deblurred_image, 0, 1, NORM_MINMAX, CV_64FC1);
    }
    Mat finalimage;
    cv::merge(deconved, finalimage);

    return finalimage;
}

cv::Mat deBlur::delta_kernel(int s){
    int c;
    Mat out;
    if (s % 2 == 0){
        s = s + 1;
    }
    out = Mat::zeros(s, s, CV_64FC1); //CHECK TYPE
    c = (int)floor(s / 2);
    out.at<double>(c, c) = 1;
    return out;
}
void displayImage(Mat image, std::string name, int channel){
    if (image.channels() != 2){
        namedWindow(name, 0);// Create a window for display.
        resizeWindow(name, 200, 200);
        imshow(name, image);
        waitKey(0);
    }
    else{
        if (image.channels() == 2){
            namedWindow(name, 0);// Create a window for display.
            resizeWindow(name, 200, 200);
            imshow(name, getChannel(image, channel));
            waitKey(0);
        }
    }
    return;
}

void displayImage(Mat image, std::string name){
    if (image.channels() != 2){
        namedWindow(name, 0);// Create a window for display.
        //resizeWindow(name,w , h);
        imshow(name, image);
        waitKey(0);
    }
    else{
        if (image.channels() == 2){
            namedWindow(name, 0);// Create a window for display.
            //resizeWindow(name, w, h);
            imshow(name, getChannel(image, 0));
            waitKey(0);
        }
    }
    return;
}

void printMat(Mat input){
    std::cout << input << std::endl;
    return;
}

void writeMat2File(Mat input, std::string filename){
    cv::FileStorage file("filename.txt", cv::FileStorage::WRITE);

    // Write to file!
    file << input;

    return;
}

Mat deBlur::estK(Mat Prediction, Mat OriginalBlurredScaledSingleChannel, Mat PSF, int numberOfIterations){
    cv::Mat  Q, P, PredictionPadded, BlurredPadded, ex, ker, corrBlur;
    Point anchor;
    cv::Size ks;
    double error = 1;
    int iteration = 1;
    //Padding-------------------------------------------------------------------------------------------------------------------------
    ks = getPadSize(PSF);
    PredictionPadded = paddarray(Prediction, ks, "replicate", "both");
    BlurredPadded = paddarray(OriginalBlurredScaledSingleChannel, ks, "replicate", "both");
    ex = PSF;
    //Get kernel while loop---------------------------------------------------------------------------------------------------------
    while ((error > 0.0001) && (iteration < numberOfIterations)){
        ker = getKMulSpectrumSupport(PredictionPadded, BlurredPadded, PSF, this->belta);   //Q:Prediction, P : Result, f : kernel
        error = norm((ker - ex), 2);
        ex = ker;

        //BlurredPadded = paddarray(OriginalBlurredScaledSingleChannel, ks, "replicate", "both");
        iteration++;
        //std::cout<<Kernel Estimation loop completed <<std::endl;
    }

    return ker;
}

void deBlur::coreDeconv(Mat OriginalBlurredScaled, Mat OriginalBlurredMinimumScaleSingleChannel, Mat PSF, int iterScaleNum, bool isFinal, Mat OriginalBlurred, double sigmaRange){
    Mat  Temp;
    Mat Prediction;
    Size TempSize;
    Temp = OriginalBlurredMinimumScaleSingleChannel;
    //Prediction = OriginalBlurredMinimumScaleSingleChannel;


    //Bilateral--------------------------------------------------------------------------
    double minValue, maxValue, delta;
    cv::minMaxLoc(Temp, &minValue, &maxValue);
    delta = maxValue - minValue;
    if (this->sigmaColor == 0){
        this->sigmaColor = 0.1*delta;
    }
    if (this->sigmaSpace == 0){
        this->sigmaSpace = min(Temp.cols, Temp.rows);
        this->sigmaSpace = this->sigmaSpace / 16.0;  //#!CAUTION#
    }


    for (int thisinnerIter = 0; thisinnerIter < iterScaleNum; thisinnerIter++){
        //Temp.convertTo(Temp, CV_32F);
        //medianBlur(Temp, Temp, 3);
        Temp.convertTo(Temp, CV_64F);
        bilateralFilter(Temp, Prediction, this->sigmaColor, this->sigmaSpace);
        normalize(Prediction, Prediction, 0, 1, NORM_MINMAX, CV_64FC1);
        Prediction = shock_filter(Prediction, this->shockFilterIter, this->shockFilterdt, 1); /////////Here!
        normalize(Prediction, Prediction, 0, 1, NORM_MINMAX, CV_64FC1);
        this->OriginalBlurredScaledSingleChannel = r2g(OriginalBlurredScaled);
        this->PSFResult = estK(Prediction, this->OriginalBlurredScaledSingleChannel, PSF, 10);
        if ((thisinnerIter != iterScaleNum) || (!isFinal)){
            this->deblurredImage = deconv(this->OriginalBlurredScaledSingleChannel, this->PSFResult, this->alpha);
            normalize(this->deblurredImage, this->deblurredImage, 0, 1, NORM_MINMAX, CV_64FC1);
            Mat Temp = this->deblurredImage;
        }

        //Final devonvolution-------------------------------------------------------------------------------

        if ((thisinnerIter == iterScaleNum - 1) && (this->isFinal)){
            this->deblurredImage = deconv(OriginalBlurred, this->PSFResult, this->alpha);

            normalize(this->deblurredImage, this->deblurredImage, 0, 1, NORM_MINMAX, CV_64FC3);
        }
    }

    return;
}

Mat conjMat(Mat src){
    Mat output = src;
    Mat theChannels[2];
    Mat * image;
    image = new Mat;
    split(src, theChannels);
    if (src.channels() == 2)
    {
        theChannels[1] = -theChannels[1];
        merge(theChannels, 2, output);
    }
    else
    {
        exit(19);
    }

    return output;
}

void deBlur::analyzeKernel(void){
    double elem = this->initsize;
    int i = 1;
    bool flag = true;
    while (flag){
        this->kernelArray.push_back((int)elem);
        elem = floor(elem / 2);
        if (elem <= 5){
            if ((int)elem % 2 == 0){
                elem++;
            }
            this->kernelArray.push_back((int)elem);
            flag = false;
        }
        if ((int)elem % 2 == 0){
            elem++;
        }
    }
    return;
}
//==============v1=====================================
Mat blinddeblurmap(string _filename = "../images/book3.jpg", int _initsize = 21, int _iterScaleNum = 10, double _alpha = 5.0, double _freqCut = 16.0, int _belta = 2, double _sigmaSpace = 0, double _sigmaColor = 0, int _shockFilterIter = 3, double _shockFilterdt = 0.5)
{
    clock_t begin = clock();
    deBlur theProcess;
    theProcess.initsize = _initsize;
    theProcess.iterScaleNum = _iterScaleNum;
    theProcess.alpha = _alpha;
    theProcess.freqCut = _freqCut;
    theProcess.belta = _belta;
    theProcess.sigmaSpace = _sigmaSpace;
    theProcess.sigmaColor = _sigmaColor;
    theProcess.shockFilterIter = _shockFilterIter;
    theProcess.shockFilterdt = _shockFilterdt;
    theProcess.filename = _filename;
    theProcess.img = imread(_filename, CV_64F);

    //theProcess.initialization();//Set initialization parameters

    theProcess.readImage();
    theProcess.analyzeKernel();
    theProcess.resizeBlurredToMinimumScale();
    theProcess.iterativeDeblurring();
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << elapsed_secs << std::endl;
    return theProcess.deblurredImage;
}

//=================v2===============================

Mat blinddeblurmap(Mat _inputImage, int _initsize = 21, int _iterScaleNum = 10, double _alpha = 5.0, double _freqCut = 16.0, int _belta = 2, double _sigmaSpace = 0, double _sigmaColor = 0, int _shockFilterIter = 3, double _shockFilterdt = 0.5)
{
    clock_t begin = clock();
    deBlur theProcess;
    theProcess.initsize = _initsize;
    theProcess.iterScaleNum = _iterScaleNum;
    theProcess.alpha = _alpha;
    theProcess.freqCut = _freqCut;
    theProcess.belta = _belta;
    theProcess.sigmaSpace = _sigmaSpace;
    theProcess.sigmaColor = _sigmaColor;
    theProcess.shockFilterIter = _shockFilterIter;
    theProcess.shockFilterdt = _shockFilterdt;
    theProcess.img = _inputImage;

    //theProcess.initialization(); //Set initialization parameters

    theProcess.readImage();
    theProcess.analyzeKernel();
    theProcess.resizeBlurredToMinimumScale();
    theProcess.iterativeDeblurring();
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << elapsed_secs << std::endl;
    return theProcess.deblurredImage;
}
//=================================================

Mat  deblur(Mat image)
{
    //#!CAUTION#

    int _initsize = 21;
    int _iterScaleNum = 10;
    double _alpha = 0.5;
    double _freqCut = 40;
    int _belta = 2;
    double _sigmaSpace = 200;
    double _sigmaColor = 0;
    int _shockFilterIter = 2;
    double _shockFilterdt = 0.5;
    Mat _out;
    Mat _input = image;

    _input.convertTo(_input, CV_32F);
    medianBlur(_input, _input, 3);
    _input.convertTo(_input, CV_64F);
    _out = blinddeblurmap(_input, _initsize, _iterScaleNum, _alpha, _freqCut, _belta, _sigmaSpace, _sigmaColor, _shockFilterIter, _shockFilterdt);

    //_out.convertTo(_out, CV_32F);
    //medianBlur(_out, _out, 3);
    //_out.convertTo(_out, CV_64F);
    //_out = blinddeblurmap(_out, _initsize, _iterScaleNum, _alpha, _freqCut, _belta, _sigmaSpace, _sigmaColor, _shockFilterIter, _shockFilterdt);

    //_out.convertTo(_out, CV_32F);
    //medianBlur(_out, _out, 3);
    //_out.convertTo(_out, CV_64F);
    //_out = blinddeblurmap(_out, _initsize, _iterScaleNum, _alpha, _freqCut, _belta, _sigmaSpace, _sigmaColor, _shockFilterIter, _shockFilterdt);

    return  _out;
}

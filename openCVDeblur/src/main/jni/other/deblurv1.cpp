#include "deblurv1.h"

using namespace std;
using namespace cv;

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





Mat getChannel(Mat input, int theChannel);
void printMat(Mat input);
void displayKernel(Mat image, string name);
void computeDFT(Mat& image, Mat& dest);
void computeIDFT(Mat& complex, Mat& dest);
void deconvolute(Mat& img, Mat& kernel);
void mulComplexMats(Mat A, Mat B, Mat &Result, int flag, bool what);
void writeMat2File(Mat input, std::string filename);
void divSpectrums(InputArray _srcA, InputArray _srcB, OutputArray _dst, int flags, bool conjB);
void shift(const Mat& src, Mat& dst, Point2f delta, int fill = BORDER_CONSTANT, Scalar value = Scalar(0, 0, 0, 0));



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
    void resizeBlurredToMinimumScale(void);
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
    Mat deconv_TV(Mat Blurred, Mat PSF, double w0alpha);
    //Tester----------------------------------------------------
    void testExample(void);
};


Mat deBlur::getKMulSpectrumSupport(Mat PredictionPadded, Mat BlurredPadded, Mat PSF, double w0belta){
    //Initialization
    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "GetK");

    double minValue, maxValue;
    Mat kernelOut;
    kernelOut = delta_kernel(PSF.size().height);

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

    //std::vector<Mat> Calc;

    PredictionPaddedIn = PredictionPadded;
    BlurredPaddedIn = BlurredPadded;

    A = fft2(PredictionPaddedIn);
    b = fft2(BlurredPaddedIn);

    for (int i = 0; i < 0; i++){

        k = psf2otf(kernelOut, A.size());



        //d=conj(A).mul(b)-(conj(A).mul(A)+belta).mul(k);=========================================
        mulSpectrums(b, A, Temp, 0, true);
        mulSpectrums(A, A, Temp2, 0, true);
        beltaChannels[0] = Mat(A.size(), CV_64F, Scalar(w0belta));
        beltaChannels[1] = Mat(A.size(), CV_64F, Scalar(0));
        merge(beltaChannels, 2, beltaMat);
        Temp2 = Temp2 + beltaMat;
        mulSpectrums(Temp2, k, Temp2, 0, false);
        d = Temp - Temp2;

        //renta = (conj(A.*d).*b - conj(A.*d).*A.*k - belta.*conj(d).*k). / (conj(A.*d).*A.*d + belta*conj(d).*d);======
        //Nominator

        Mat Product1 = Mat::zeros(PredictionPadded.size(), CV_64F);
        Mat Product2 = Mat::zeros(PredictionPadded.size(), CV_64F);
        Mat Product3 = Mat::zeros(PredictionPadded.size(), CV_64F);
        Mat Product4 = Mat::zeros(PredictionPadded.size(), CV_64F);
        Mat Product5 = Mat::zeros(PredictionPadded.size(), CV_64F);
        Mat Product6 = Mat::zeros(PredictionPadded.size(), CV_64F);
        Mat Product7 = Mat::zeros(PredictionPadded.size(), CV_64F);
        Mat Product8 = Mat::zeros(PredictionPadded.size(), CV_64F);
        Mat Product9 = Mat::zeros(PredictionPadded.size(), CV_64F);


        //0
        mulSpectrums(A, d, Product1, 0, false);//mulSpectrums
        mulSpectrums(b, Product1, Product2, 0, true);
        //1
        mulSpectrums(A, d, Product1, 0, false);//mulSpectrums
        mulSpectrums(A, Product1, Product3, 0, true);
        //2
        mulSpectrums(k, d, Product5, 0, true);
        Product6 = w0belta* Product5;

        //Denom . / (conj(A.*d).*A.*d + belta*conj(d).*d);========================================
        //3
        mulSpectrums(A, d, Product1, 0, false);
        mulSpectrums(A, Product1, Product3, 0, true);
        mulSpectrums(Product3, d, Product7, 0, false);
        //4
        mulSpectrums(d, d, Product8, 0, true);
        Product9 = w0belta*Product8;

        Mat theNom = Product2 - Product3 - Product6;
        Mat theDeNom = Product7 + Product9;
        divSpectrums(theNom, theDeNom, renta2, 0, false);
        mulSpectrums(renta2, d, renta2, 0, false);
        k = k + renta2;
        kernelOut = otf2psf(k, PSF.size());
        kernelOut = abs(kernelOut);
        minMaxLoc(kernelOut, &minValue, &maxValue);

        for (int i = 0; i < kernelOut.rows; i++){
            for (int j = 0; j < kernelOut.cols; j++){
                if (kernelOut.at<double>(i, j) < (maxValue / this->freqCut)){
                    kernelOut.at<double>(i, j) = 0;
                }

            }
        }

        double s = sum(kernelOut)[0];
        //kernelOut = ((double)1.0 / ((double)(maxValue)))*kernelOut;
        kernelOut = ((1.0/(s)))*kernelOut;
        //kernelOut = ((1.0 / (s*s)))*kernelOut;							//#!CAUTION#//
        //std::cout << ((1.0 / (s*s*s*s))) << std::endl;
        //kernelOut = 1 * kernelOut;
    }

    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "GetK:maxValue: %f",maxValue);
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
        dft(input, fft, DFT_COMPLEX_OUTPUT, input.rows);
    }
    else{
        exit(37);
    }
    return fft;
}

Mat deBlur::ifft2(Mat input){
    Mat inputMat = input;
    Mat inverseTransform = Mat::zeros(input.size(), CV_64FC1);

    if (inputMat.channels() == 2){

        dft(inputMat, inverseTransform, DFT_INVERSE | DFT_SCALE | DFT_REAL_OUTPUT, input.rows);

    }
    else{
        exit(37);
    }
    return inverseTransform;
}



Mat deBlur::r2g(Mat input){
    Mat output;
    if (input.channels() >1 ){
        vector<Mat> channels(3);
        split(input, channels);
        output = ((double)1 / (double)3)*(channels[0] + channels[1] + channels[2]);
        output.convertTo(output, CV_64FC1);
    }
    else
    {
        output = input;
        output.convertTo(output, CV_64FC1);
    }

    return output;

/*
    input.convertTo(input,CV_32F);
    Mat output=Mat::zeros(input.size(),CV_32F);
    if (input.channels() >1){
        cvtColor(input, output, CV_BGR2GRAY);
    }
    else{
        output = input;

    }
    output.convertTo(output, CV_64F);
    return output;
    */
}



Size deBlur::getPadSize(Mat f){
    return Size(((f.rows - 1) / 2), ((f.cols - 1) / 2));
}

Mat deBlur::paddarray(Mat input, Size padding, std::string method, std::string direction){
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
    B.convertTo(B, CV_64FC3);
    B = (1.0 / 255.0)*B;
    this->OriginalBlurred = B;
    return;
}



Mat deBlur::shock_filter(Mat IO, int iter, double _dt, double h){
    Mat I_mx ;
    Mat I_px ;
    Mat I_my ;
    Mat I_py;
    Mat I_x;
    Mat I_y;
    Mat Dx;
    Mat Dy;
    Mat I_mx_abs;
    Mat I_px_abs;
    Mat I_my_abs;
    Mat I_py_abs;
    Mat I_xx;
    Mat I_yy;
    Mat I_xy;
    Mat I_xy_m_dy;
    Mat I_xy_p_dy;
    Mat I_xy_my;
    Mat I_xy_py;
    Mat a_grad_I;
    Mat I_nn;
    Mat I_ee;
    Mat I_buf;
    Mat I_buf2;
    Mat I_buf3;
    Mat a2_grad_I;
    Mat result;
    Mat I_t;
    Mat I_in;
    Mat kernelx;
    Mat kernely;
    Mat IShiftColsForward;
    Mat IShiftColsBackwards;
    Mat IShiftRowsForward;
    Mat IShiftRowsBackwards;
    Mat IxShiftRowsForward;
    Mat IxShiftRowsBackwards;

    I_mx =Mat::zeros(IO.size(),CV_32F);
    I_px =Mat::zeros(IO.size(),CV_32F);
    I_my =Mat::zeros(IO.size(),CV_32F);
    I_py=Mat::zeros(IO.size(),CV_32F);
    I_x=Mat::zeros(IO.size(),CV_32F);
    I_y=Mat::zeros(IO.size(),CV_32F);
    Dx=Mat::zeros(IO.size(),CV_32F);
    Dy=Mat::zeros(IO.size(),CV_32F);
    I_mx_abs=Mat::zeros(IO.size(),CV_32F);
    I_px_abs=Mat::zeros(IO.size(),CV_32F);
    I_my_abs=Mat::zeros(IO.size(),CV_32F);
    I_py_abs=Mat::zeros(IO.size(),CV_32F);
    I_xx=Mat::zeros(IO.size(),CV_32F);
    I_yy=Mat::zeros(IO.size(),CV_32F);
    I_xy=Mat::zeros(IO.size(),CV_32F);
    I_xy_m_dy=Mat::zeros(IO.size(),CV_32F);
    I_xy_p_dy=Mat::zeros(IO.size(),CV_32F);
    I_xy_my=Mat::zeros(IO.size(),CV_32F);
    I_xy_py=Mat::zeros(IO.size(),CV_32F);
    a_grad_I=Mat::zeros(IO.size(),CV_32F);
    I_nn=Mat::zeros(IO.size(),CV_32F);
    I_ee=Mat::zeros(IO.size(),CV_32F);
    I_buf=Mat::zeros(IO.size(),CV_32F);
    I_buf2=Mat::zeros(IO.size(),CV_32F);
    I_buf3=Mat::zeros(IO.size(),CV_32F);
    a2_grad_I=Mat::zeros(IO.size(),CV_32F);
    result=Mat::zeros(IO.size(),CV_32F);
    I_t=Mat::zeros(IO.size(),CV_32F);
    I_in=Mat::zeros(IO.size(),CV_32F);
    kernelx=Mat::zeros(IO.size(),CV_32F);
    kernely=Mat::zeros(IO.size(),CV_32F);
    IShiftColsForward=Mat::zeros(IO.size(),CV_32F);
    IShiftColsBackwards=Mat::zeros(IO.size(),CV_32F);
    IShiftRowsForward=Mat::zeros(IO.size(),CV_32F);
    IShiftRowsBackwards=Mat::zeros(IO.size(),CV_32F);
    IxShiftRowsForward=Mat::zeros(IO.size(),CV_32F);
    IxShiftRowsBackwards=Mat::zeros(IO.size(),CV_32F);


    int nx = IO.rows;
    int ny = IO.cols;
    double h_2 = pow(h, 2);

    I_in = IO;
    I_in.convertTo(I_in,CV_32F);
    for (int i = 0; i < iter; i++){


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
                if (I_mx.at<float>(i, j)*I_px.at<float>(i, j) < 0){
                    Dx.at<float>(i, j) = 0;
                }
            }
        }
        for (int i = 0; i < Dy.rows; i++){
            for (int j = 0; j < Dy.cols; j++){
                if (I_my.at<float>(i, j)*I_py.at<float>(i, j) < 0){
                    Dy.at<float>(i, j) = 0;
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
                if (a2_grad_I.at<float>(i, j) == 0){
                    I_nn.at<float>(i, j) = I_xx.at<float>(i, j);
                    I_ee.at<float>(i, j) = I_yy.at<float>(i, j);
                }
            }
        }

        I_t = Mat::zeros(IO.size(), CV_32F);
        //I_t = a2_grad_I / h;
        //displayImage(I_t, "");
        int thesign;
        for (int i = 0; i < I_t.rows; i++){
            for (int j = 0; j < I_t.cols; j++){
                if (I_nn.at<float>(i, j) > 0){
                    thesign = 1;
                }
                else{
                    thesign = -1;
                }
                I_t.at<float>(i, j) = -thesign*a2_grad_I.at<float>(i, j) / (float)h;
            }
        }

        I_in = I_in + _dt *I_t;
    }

    I_in.convertTo(I_in,CV_64F);

    return I_in;
}


void deBlur::iterativeDeblurring(void){
    int currentIterationFactor = 0;
    double denominator;
    double resizeFactor;
    Mat InitialPSF;
    Mat temp;
    int InitialPSFSize;
    //Start iterations
    this->isFinal = false;

    for (int thisIter = 0; thisIter < this->scales; thisIter++){
        //Check of is final-----------------------------------------------
        if (thisIter == this->scales - 1){
            this->isFinal = true;
        }
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


            resize(this->deblurredImage, this->OriginalBlurredScaledMinSingleChannel, this->deblurredImage.size() + this->deblurredImage.size(), 0, 0, CV_INTER_CUBIC);

            //OriginalBlurredScaledMinSingleChannel = mycubic_2();
            //resize(this->OriginalBlurredScaledSingleChannel, this->OriginalBlurredScaledMinSingleChannel, this->deblurredImage.size() + this->deblurredImage.size(), 0, 0, CV_INTER_CUBIC);


        }

    }



    return;
}




void shift(const Mat& src, Mat& dst, Point2f delta, int fill, Scalar value) {

    // error checking
    assert(fabs(delta.x) < src.cols && fabs(delta.y) < src.rows);

    // split the shift into integer and subpixel components
    Point2i deltai((int)ceil(delta.x), (int)ceil(delta.y));
    Point2f deltasub(fabs(delta.x - deltai.x), fabs(delta.y - deltai.y));

    // INTEGER SHIFT
    // first create a border around the parts of the Mat that will be exposed
    int t = 0, b = 0, l = 0, r = 0;
    if (deltai.x > 0) l = deltai.x;
    if (deltai.x < 0) r = -deltai.x;
    if (deltai.y > 0) t = deltai.y;
    if (deltai.y < 0) b = -deltai.y;
    Mat padded;
    copyMakeBorder(src, padded, t, b, l, r, fill, value);

    // SUBPIXEL SHIFT
    float eps = std::numeric_limits<float>::epsilon();
    if (deltasub.x > eps || deltasub.y > eps) {
        switch (src.depth()) {
            case CV_32F:
            {
                Matx<float, 1, 2> dx(1 - deltasub.x, deltasub.x);
                Matx<float, 2, 1> dy(1 - deltasub.y, deltasub.y);
                sepFilter2D(padded, padded, -1, dx, dy, Point(0, 0), 0, BORDER_CONSTANT);
                break;
            }
            case CV_64F:
            {
                Matx<double, 1, 2> dx(1 - deltasub.x, deltasub.x);
                Matx<double, 2, 1> dy(1 - deltasub.y, deltasub.y);
                sepFilter2D(padded, padded, -1, dx, dy, Point(0, 0), 0,BORDER_CONSTANT);
                break;
            }
            default:
            {
                Matx<float, 1, 2> dx(1 - deltasub.x, deltasub.x);
                Matx<float, 2, 1> dy(1 - deltasub.y, deltasub.y);
                padded.convertTo(padded, CV_32F);
                sepFilter2D(padded, padded, CV_32F, dx, dy, Point(0, 0), 0, BORDER_CONSTANT);
                break;
            }
        }
    }

    // construct the region of interest around the new matrix
    Rect roi = Rect(std::max(-deltai.x, 0), std::max(-deltai.y, 0), 0, 0) + src.size();
    dst = padded(roi);
}


void deBlur::resizeBlurredToMinimumScale(void){
    this->scales = this->kernelArray.size();
    this->scalesDenominator = pow((scales - 1), 2);
    this->scalesFactor = 1 / scalesDenominator;
    Size s = this->OriginalBlurred.size();
    Size s2 = Size((int)((double)s.height * scalesFactor), (int)((double)s.width * scalesFactor));
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
        cv::Rect myROI(ks.width, ks.height, width, height);

        deblurred_image = theResult(myROI);
        deconved.push_back(deblurred_image);
        normalize(deblurred_image, deblurred_image, 0, 1, NORM_MINMAX, CV_64FC1);
    }
    Mat finalimage;
    cv::merge(deconved, finalimage);

    return finalimage;
}


Mat deBlur::delta_kernel(int s){
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

Mat deBlur::estK(Mat Prediction, Mat OriginalBlurredScaledSingleChannel, Mat PSF, int numberOfIterations){
    Mat PredictionPadded;
    Mat BlurredPadded;
    Mat ker;
    Mat ex;
    Size ks;
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
    }


    return ker;
}

void deBlur::coreDeconv(Mat OriginalBlurredScaled, Mat OriginalBlurredMinimumScaleSingleChannel, Mat PSF, int iterScaleNum, bool isFinal, Mat OriginalBlurred, double sigmaRange){

    Mat Temp=OriginalBlurredMinimumScaleSingleChannel;
    Mat Prediction=OriginalBlurredMinimumScaleSingleChannel;



    for (int thisinnerIter = 0; thisinnerIter < iterScaleNum; thisinnerIter++){






        //PREDICTION
        //Temp.convertTo(Temp, CV_32F);
        //medianBlur(Temp, Temp, 3);
        Temp.convertTo(Prediction, CV_64F);

        //BILATERAL FILTER SETTINGS
        double minValue, maxValue, delta;
        cv::minMaxLoc(Prediction, &minValue, &maxValue);
        delta = maxValue - minValue;
        __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "delta:%f", delta);
        if (this->sigmaColor == 0){
            this->sigmaColor = 0.1*delta;
        }
        if (this->sigmaSpace == 0){
            this->sigmaSpace = min(Prediction.cols, Prediction.rows);
            this->sigmaSpace = this->sigmaSpace / 16.0;  //#!CAUTION#
        }

        cv_extend::bilateralFilter(Prediction, Prediction, this->sigmaColor, this->sigmaSpace);
        //normalize(Prediction, Prediction, 0, 1, NORM_MINMAX, CV_64FC1);
        Prediction = shock_filter(Prediction, this->shockFilterIter, pow(this->shockFilterdt,((double)thisinnerIter)), 1); //#Caution Here!
        //Prediction = shock_filter(Prediction, this->shockFilterIter, this->shockFilterdt,1); //#Caution Here!





          //ESTK
          this->OriginalBlurredScaledSingleChannel = r2g(OriginalBlurredScaled);
          this->PSFResult = estK(Prediction, this->OriginalBlurredScaledSingleChannel, PSF, 10);

          //DECONVOLUTION
          if ((thisinnerIter != iterScaleNum) || (!isFinal)){

              this->deblurredImage = deconv(this->OriginalBlurredScaledSingleChannel, this->PSFResult, this->alpha);
              //normalize(this->deblurredImage, this->deblurredImage, 0, 1, NORM_MINMAX, CV_64FC1);
              Mat Temp = this->deblurredImage;


          }

          //FINALDECONVOLUTION-------------------------------------------------------------------------------

          if ((thisinnerIter == iterScaleNum - 1) && (this->isFinal)){

              this->deblurredImage = deconv(OriginalBlurred, this->PSFResult,  this->alpha);
              //normalize(this->deblurredImage, this->deblurredImage, 0, 1, NORM_MINMAX, CV_64FC3);

          }


    }
    //this->deblurredImage=Prediction;
    return;
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



Mat blinddeblurmap(Mat _inputImage, int _initsize, int _iterScaleNum, double _alpha, double _freqCut, int _belta, double _sigmaSpace, double _sigmaColor, int _shockFilterIter, double _shockFilterdt)
{
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


    theProcess.readImage();
    theProcess.analyzeKernel();
    theProcess.resizeBlurredToMinimumScale();
    theProcess.iterativeDeblurring();
    theProcess.deblurredImage=255*theProcess.deblurredImage;
    return theProcess.deblurredImage;
}






Mat  deblur(Mat image)
{
    //#!CAUTION#
    int _initsize = 21;
    int _iterScaleNum = 10;
    double _alpha = 50;
    double _freqCut = 16.0;
    int _belta = 2;
    double _sigmaSpace = 0;
    double _sigmaColor = 0;
    int _shockFilterIter = 1;
    double _shockFilterdt = 0.7;
    Mat _out;
    Mat _input = image;

    //_input.convertTo(_input, CV_32FC3);
    medianBlur(_input, _input, 3);
    //_input.convertTo(_input, CV_64FC3);


    _out = blinddeblurmap(_input, _initsize, _iterScaleNum, _alpha, _freqCut, _belta, _sigmaSpace, _sigmaColor, _shockFilterIter, _shockFilterdt);

    //_out.convertTo(_out, CV_32F);
    //medianBlur(_out, _out, 3);
    //_out.convertTo(_out, CV_64F);
    //_out = blinddeblurmap(_out, _initsize, _iterScaleNum, _alpha, _freqCut, _belta, _sigmaSpace, _sigmaColor, _shockFilterIter, _shockFilterdt);
    //_out.convertTo(_out, CV_32F);
    //medianBlur(_out, _out, 3);
    //_out.convertTo(_out, CV_64F);
    //_out = blinddeblurmap(_out, _initsize, _iterScaleNum, _alpha, _freqCut, _belta, _sigmaSpace, _sigmaColor, _shockFilterIter, _shockFilterdt);

    _out.convertTo(_out, CV_8UC3);
    return  _out;

}


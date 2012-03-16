#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

#define BAYESIAN_NUMBER_NEAREST   200
#define BAYESIAN_SIGMA            8.f
#define BAYESIAN_SIGMA_C          5.f
#define BAYESIAN_MAX_CLUS           3

class BayesianMatting
{
public:
    BayesianMatting(Mat img, Mat trimap);
    ~BayesianMatting();
    
    void setParameters(int n=BAYESIAN_NUMBER_NEAREST, double sigma=BAYESIAN_SIGMA, double sigmaC=BAYESIAN_SIGMA_C);
    double solve();
    
private:
    void initialize();
    vector<Point> getContour(Mat trimap);
    void CollectSampleSet(int x, int y, vector<pair<cv::Point, float> > &fg_set, vector<pair<cv::Point, float> > &bg_set);
    void GetGMMModel(int x, int y, vector<float> &fg_weight, const vector<Mat> fg_mean, vector<Mat> inv_fg_cov, vector<float> &bg_weight, vector<Mat> bg_mean, vector<Mat> inv_bg_cov);
    void CalculateNonNormalizeCov(Mat cImg, vector<pair<cv::Point, float> > &clus_set, Mat mean, Mat cov);
    void CalculateMeanCov(Mat cImg, vector<pair<cv::Point, float> > &clus_set, Mat mean, Mat cov);
    void CalculateWeightMeanCov(Mat cImg, vector<pair<cv::Point, float> > &clus_set, float &weight, Mat mean, Mat cov);
    
    Mat img, fgImg, bgImg;
    Mat maskFg, maskBg, maskUnknown, maskUnsolved;
    Mat trimap, alphamap;
    int nearest;
    double sigma, sigmaC;
};

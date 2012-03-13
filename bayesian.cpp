#include "bayesian.h"

BayesianMatting::BayesianMatting(cv::Mat img, cv::Mat trimap)
{
    this->img = img.clone();
    
    // Convert the trimap into a single channel image
    if(trimap.channels()==3)
    {
        this->trimap = Mat(img.size(), CV_8UC1);
        cv::cvtColor(trimap, this->trimap, CV_BGR2GRAY, 1);
    }
    else
    {
        this->trimap = trimap.clone();
    }
    
    initialize();
    setParameters();
}

BayesianMatting::~BayesianMatting()
{

}

void BayesianMatting::initialize()
{
    cv::Size imgSize = trimap.size();
    
    fgImg = Mat(imgSize, CV_8UC3, cv::Scalar(0, 0, 0));
    bgImg = Mat(imgSize, CV_8UC3, cv::Scalar(0, 0, 0));
    maskFg = Mat(imgSize, CV_8UC1, cv::Scalar(0));
    maskBg = Mat(imgSize, CV_8UC1, cv::Scalar(0));
    maskUnknown = Mat(imgSize, CV_8UC1, cv::Scalar(0));
    alphamap = Mat(imgSize, CV_32FC1, cv::Scalar(0));
    
    for(int y=0;y<imgSize.height;y++)
    {
        for(int x=0;x<imgSize.width;x++)
        {
            uchar v = trimap.at<uchar>(y, x);
            if(v==0)
                maskBg.at<uchar>(y, x) = 255;
            else if(v==255)
                maskFg.at<uchar>(y, x) = 255;
            else
                maskUnknown.at<uchar>(y, x) = 255;
        }
    }
    
    img.copyTo(fgImg, maskFg);
    img.copyTo(bgImg, maskBg);
    
    return;
}

vector<cv::Point> BayesianMatting::getContour(Mat img)
{
    vector<cv::Point> ret;
    vector<vector<cv::Point> > contours;
    
    cv::findContours(img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    int numContours = contours.size();
    for(int i=0;i<numContours;i++)
    {
        int count = contours[i].size();
        for(int j=0;j<count;j++)
        {
            ret.push_back(contours[i][j]);
        }
    }
    
    return ret;
}

void BayesianMatting::setParameters(int N, double sigma, double sigmaC)
{
    this->nearest = N;
    this->sigma = sigma;
    this->sigmaC = sigmaC;
}

double BayesianMatting::solve()
{
    vector<cv::Point> p = getContour(maskUnknown);
    for(int i=0;i<p.size();i++)
    {
        cv::circle(img, p[i], 1, CV_RGB(255,0,0), 1);
    }
    
    cv::imshow("img", img);
    cv::waitKey(0);
}

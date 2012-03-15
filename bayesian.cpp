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
    maskUnsolved = Mat(imgSize, CV_8UC1, cv::Scalar(0));
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
    maskUnknown.copyTo(maskUnsolved);
    
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

void BayesianMatting::CollectSampleSet(int x, int y, vector<pair<cv::Point, float> > &fg_set, vector<pair<cv::Point, float> > &bg_set)
{
    // Erase any existing set
    fg_set.clear();
    bg_set.clear();
    
    #define UNSURE_DIST 1
    pair<cv::Point, float> sample;
    float dist_weight;
    float inv_2sigma_square = 1.0/(2.0*this->sigma*this->sigma);
    int dist=1;
    
    while(fg_set.size()<nearest)
    {
        if(y-dist>=0)
        {
            for(int z=max(0, x-dist);z<=min(this->img.cols-1, x+dist);z++)
            {
                dist_weight = expf(-(dist*dist+(z-x)*(z-x)) * inv_2sigma_square);
                
                // We know this pixel belongs to the foreground
                if(maskFg.at<uchar>(y-dist, z)!=0)
                {
                    sample.first.x = z;
                    sample.first.y = y-dist;
                    sample.second = dist_weight;
                    
                    fg_set.push_back(sample);
                    if(fg_set.size()==nearest)
                        goto BG;
                }
                else if(dist<UNSURE_DIST && maskUnknown.at<uchar>(y-dist, z)!=0 && maskUnsolved.at<uchar>(y-dist, z)==0)
                {
                    sample.first.x = z;
                    sample.first.y = y-dist;
                    
                    float alpha = alphamap.at<float>(y-dist, z);
                    sample.second = dist_weight*alpha*alpha;                
                    
                    fg_set.push_back(sample);
                    if(fg_set.size()==nearest)
                        goto BG;
                }
            }
        }
        
        if(y+dist<=img.rows-1)
        {
            for(int z=max(0, x-dist+1);z<=min(img.cols-1, x+dist);z++)
            {
                dist_weight = expf(-(dist*dist+(z-x)*(z-x)) * inv_2sigma_square);
                
                if(maskFg.at<uchar>(y+dist, z)!=0)
                {
                    sample.first.y = y+dist;
                    sample.first.x = z;
                    sample.second = dist_weight;
                    
                    fg_set.push_back(sample);
                    if(fg_set.size()==nearest)
                        goto BG;
                }
                else if(dist<UNSURE_DIST && maskUnknown.at<uchar>(y+dist, x)!=0 && maskUnsolved.at<uchar>(y+dist, z)==0)
                {
                    sample.first.x = z;
                    sample.first.y = y+dist;
                    
                    float alpha = alphamap.at<float>(y+dist, z);
                    sample.second = dist_weight*alpha*alpha;
                    
                    fg_set.push_back(sample);
                    if(fg_set.size()==nearest)
                        goto BG;
                }
            }
        }
        
        if(x-dist>=0)
        {
            for(int z=max(0, y-dist+1);z<=min(img.rows-1, y+dist-1); z++)
            {
                dist_weight = expf(-(dist*dist+(z-y)*(z-y)) * inv_2sigma_square);
                
                if(maskFg.at<uchar>(z, x-dist)!=0)
                {
                    sample.first.x = x-dist;
                    sample.first.y = z;
                    sample.second = dist_weight;
                    
                    fg_set.push_back(sample);
                    if(fg_set.size()==nearest)
                        goto BG;
                }
                else if(dist<UNSURE_DIST && maskUnknown.at<uchar>(z, x-dist)!=0 && maskUnsolved.at<uchar>(z, x-dist)==0)
                {
                    sample.first.x = x-dist;
                    sample.first.y = z;
                    
                    float alpha = alphamap.at<float>(z, x-dist);
                    sample.second = dist_weight*alpha*alpha;
                    
                    fg_set.push_back(sample);
                    if(fg_set.size()==nearest)
                        goto BG;
                }
            }
        }
        
        if(x+dist<img.cols)
        {
            for(int z=max(0, y-dist+1);z<=min(img.rows-1, y+dist-1); z++)
            {
                dist_weight = expf(-(dist*dist+(y-z)*(y-z)) * inv_2sigma_square);
                
                if(maskFg.at<uchar>(z, x+dist)!=0)
                {
                    sample.first.x = x+dist;
                    sample.first.y = z;
                    sample.second = dist_weight;
                    
                    fg_set.push_back(sample);
                    if(fg_set.size()==nearest)
                        goto BG;
                }
                else if(dist<UNSURE_DIST && maskUnknown.at<uchar>(z, x+dist)!=0 && maskUnsolved.at<uchar>(z, x+dist)==0)
                {
                    sample.first.x = x+dist;
                    sample.first.y = z;
                    
                    float alpha = alphamap.at<float>(z, x+dist);
                    sample.second = dist_weight*alpha*alpha;
                    
                    fg_set.push_back(sample);
                    if(fg_set.size()==nearest)
                        goto BG;
                }
            }
        }
        
        ++dist;
    }
    
BG:
    int bg_unsure=0;
    dist=1;

    while(bg_set.size()<nearest)
    {
        if(y-dist>=0)
        {
            for(int z=max(0, x-dist);z<=min(x+dist, img.cols-1);z++)
            {
                dist_weight = expf(-(dist*dist+(z-x)*(z-x))*inv_2sigma_square);
                if(maskBg.at<uchar>(y-dist, z)!=0)
                {
                    sample.first.x = z;
                    sample.first.y = y-dist;
                    sample.second = dist_weight;
                    
                    bg_set.push_back(sample);
                    if(bg_set.size()==nearest)
                        goto DONE;
                }
                else if(dist<UNSURE_DIST && maskUnknown.at<uchar>(y-dist, z)!=0 && maskUnsolved.at<uchar>(y-dist, z)==0)
                {
                    sample.first.x = z;
                    sample.first.y = y-dist;
                    
                    float alpha = alphamap.at<float>(y-dist, z);
                    sample.second = dist_weight*alpha*alpha;
                    
                    bg_set.push_back(sample);
                    if(bg_set.size()==nearest)
                        goto DONE;
                }
            }
        }
        
        if(y+dist<img.rows)
        {
            for(int z=max(0, x-dist);z<=min(x+dist, img.cols-1);z++)
            {
                dist_weight = expf(-(dist*dist+(z-x)*(z-x))*inv_2sigma_square);
                if(maskBg.at<uchar>(y+dist, z)!=0)
                {
                    sample.first.x = z;
                    sample.first.y = y+dist;
                    sample.second = dist_weight;
                    
                    bg_set.push_back(sample);
                    if(bg_set.size()==nearest)
                        goto DONE;
                }
                else if(dist<UNSURE_DIST && maskUnknown.at<uchar>(y+dist, z)!=0 && maskUnsolved.at<uchar>(y-dist, z)==0)
                {
                    sample.first.x = z;
                    sample.first.y = y+dist;
                    
                    float alpha = alphamap.at<float>(y+dist, z);
                    sample.second = dist_weight*alpha*alpha;
                    
                    bg_set.push_back(sample);
                    if(bg_set.size()==nearest)
                        goto DONE;
                }
            }
        }
        
        if(x-dist>=0)
        {
            for(int z=max(0, y-dist+1);z<=min(y+dist-1, img.rows-1);z++)
            {
                dist_weight = expf(-(dist*dist+(y-z)*(y-z))*inv_2sigma_square);
                if(maskBg.at<uchar>(z, x-dist)!=0)
                {
                    sample.first.x = x-dist;
                    sample.first.y = z;
                    sample.second = dist_weight;
                    
                    bg_set.push_back(sample);
                    if(bg_set.size()==nearest)
                        goto DONE;
                }
                else if(dist<UNSURE_DIST && maskUnknown.at<uchar>(z, x-dist)!=0 && maskUnsolved.at<uchar>(z, x-dist)==0)
                {
                    sample.first.x = x-dist;
                    sample.first.y = z;
                    
                    float alpha = alphamap.at<float>(z, x-dist);
                    sample.second = dist_weight*alpha*alpha;
                    
                    bg_set.push_back(sample);
                    if(bg_set.size()==nearest)
                        goto DONE;
                }
            }
        }
        
        if(x+dist<img.cols)
        {
            for(int z=max(0, y-dist+1);z<=min(y+dist-1, img.rows-1);z++)
            {
                dist_weight = expf(-(dist*dist+(y-z)*(y-z))*inv_2sigma_square);
                
                if(maskBg.at<uchar>(z, x+dist)!=0)
                {
                    sample.first.x = x+dist;
                    sample.first.y = z;
                    sample.second = dist_weight;
                    
                    bg_set.push_back(sample);
                    if(bg_set.size()==nearest)
                        goto DONE;
                }
                else if(dist<UNSURE_DIST && maskUnknown.at<uchar>(z, x+dist)!=0 && maskUnsolved.at<uchar>(z, x+dist)==0)
                {
                    sample.first.x = x+dist;
                    sample.first.y = z;
                    
                    float alpha = alphamap.at<float>(z, x+dist);
                    sample.second = dist_weight*alpha*alpha;
                    
                    bg_set.push_back(sample);
                    if(bg_set.size()==nearest)
                        goto DONE;
                }
            }
        }
        
        dist++;
    }
    
DONE:
    return;
}

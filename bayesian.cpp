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
        cv::circle(img, p[i], 0, CV_RGB(255,0,0), 1);
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

void BayesianMatting::GetGMMModel(int x, int y, vector<float> &fg_weight, const vector<Mat> fg_mean, vector<Mat> inv_fg_cov, vector<float> &bg_weight, vector<Mat> bg_mean, vector<Mat> inv_bg_cov)
{
    vector<pair<cv::Point, float> > fg_set, bg_set;
    CollectSampleSet(x, y, fg_set, bg_set);
    
    Mat mean = Mat(3, 1, CV_32FC1);
    Mat cov  = Mat(3, 3, CV_32FC1);
    Mat inv_cov = Mat(3, 3, CV_32FC1);
    Mat eigval = Mat(3, 1, CV_32FC1);
    Mat eigvec = Mat(3, 3, CV_32FC1);
    Mat cur_color = Mat(3, 1, CV_32FC1);
    Mat max_eigvec = Mat(3, 1, CV_32FC1);
    Mat target_color = Mat(3, 1, CV_32FC1);
    
    vector<pair<cv::Point, float> > clus_set[BAYESIAN_MAX_CLUS];
    int nClus = 1;
    clus_set[0] = fg_set;
    
    while(nClus < BAYESIAN_MAX_CLUS)
    {
        // Find the largest eigen value
        double max_eigval = 0;
        int max_idx = 0;
        for(int i=0;i<nClus;i++)
        {
            CalculateNonNormalizeCov(fgImg, clus_set[i], mean, cov);
            
            // cov = source
            // eigval = result
            // eigvec = left orthogonal matrix
            // cov = eigvec * eigval * V
            cv::SVD svd;
            svd(cov);
            eigval = svd.w;
            eigvec = svd.u;
            //cvSVD(cov, eigval, eigvec);
            
            float temp = eigval.at<float>(0, 0);
            if(temp > max_eigval)
            {
                max_eigvec = eigvec.col(0);
                max_idx = i;
            }
        }
        
        // Split
        vector<pair<cv::Point, float> > new_clus_set[2];
        CalculateMeanCov(fgImg, clus_set[max_idx], mean, cov);
        double boundary = mean.dot(max_eigvec);
        for(size_t i=0;i<clus_set[max_idx].size();i++)
        {
            for(int j=0;j<3;j++)
                cur_color.at<float>(j, 0) = fgImg.at<cv::Vec3b>(clus_set[max_idx][i].first.y, clus_set[max_idx][i].first.x)[j];
                
                if(cur_color.dot(max_eigvec)>boundary)
                    new_clus_set[0].push_back(clus_set[max_idx][i]);
                else
                    new_clus_set[1].push_back(clus_set[max_idx][i]);
        }
        
        clus_set[max_idx] = new_clus_set[0];
        clus_set[nClus] = new_clus_set[1];
        
        nClus+=1;
    }
        
    float weight_sum, inv_weight_sum;
    weight_sum = 0;
    for(int i=0;i<nClus;i++)
    {
        CalculateWeightMeanCov(fgImg, clus_set[i], fg_weight[i], fg_mean[i], cov);
        inv_fg_cov[i] = cov.inv();
        weight_sum += fg_weight[i];
    }
        
    // Normalize weight
    inv_weight_sum = 1.0f/weight_sum;
    for(int i=0;i<nClus;i++)
        fg_weight[i] *= inv_weight_sum;
            
    // Background
    nClus = 1;
    for(int i=0;i<BAYESIAN_MAX_CLUS;i++)
        clus_set[i].clear();
    
    clus_set[0] = bg_set;
    while(nClus<BAYESIAN_MAX_CLUS)
    {
        // Find the largest eigenvalue
        double max_eigval = 0;
        int max_idx = 0;
        
        for(int i=0;i<nClus;i++)
        {
            CalculateNonNormalizeCov(bgImg, clus_set[i], mean, cov);
            
            // Compute the eigval and eigvec
            cv::SVD svd;
            svd(cov);
            eigval = svd.w;
            eigvec = svd.u;
            float temp = eigval.at<float>(0, 0);
            if(temp > max_eigval)
            {
                max_eigvec = eigvec.col(0);
                max_eigval = temp;
                max_idx = i;
            }
        }
        
        // split
        vector<pair<cv::Point, float> > new_clus_set[2];
        CalculateMeanCov(bgImg, clus_set[max_idx], mean, cov);
        double boundary = mean.dot(max_eigvec);
        for(size_t i=0;i<clus_set[max_idx].size();i++)
        {
            for(int j=0;j<3;j++)
                cur_color.at<float>(j, 0) = bgImg.at<cv::Vec3b>(clus_set[max_idx][i].first.y, clus_set[max_idx][i].first.x)[j];
            
            if(cur_color.dot(max_eigvec)>boundary)
                new_clus_set[0].push_back(clus_set[max_idx][i]);
            else
                new_clus_set[1].push_back(clus_set[max_idx][i]);
        }
        
        clus_set[max_idx] = new_clus_set[0];
        clus_set[nClus] = new_clus_set[1];
        
        nClus += 1;
    }
    
    // Return all the mean and cov for the background
    weight_sum = 0;
    for(int i=0;i<nClus;i++)
    {
        CalculateWeightMeanCov(bgImg, clus_set[i], bg_weight[i], bg_mean[i], cov);
        inv_bg_cov[i] = cov.inv();
        weight_sum += bg_weight[i];
    }
    
    // Normalize weights
    inv_weight_sum = 1.0f/weight_sum;
    for(int i=0;i<nClus;i++)
        bg_weight[i] *= inv_weight_sum;
}

void BayesianMatting::CalculateNonNormalizeCov(Mat cImg, vector<pair<cv::Point, float> > &clus_set, Mat mean, Mat cov)
{
    int cur_x, cur_y;
    float cur_w, total_w=0;
    
    Mat(Mat::zeros(mean.rows, mean.cols, CV_32FC1)).copyTo(mean);
    Mat(Mat::zeros(cov.rows, cov.cols, CV_32FC1)).copyTo(cov);
    
    for(size_t j=0;j<clus_set.size();j++)
    {
        cur_x = clus_set[j].first.x;
        cur_y = clus_set[j].first.y;
        cur_w = clus_set[j].second;
        
        for(int h=0;h<3;j++)
        {
            cv::Vec3b color = cImg.at<cv::Vec3b>(cur_y, cur_x);
            mean.at<float>(h, 0) = mean.at<float>(h, 0) + cur_w*color[h];
            for(int k=0;k<3;k++)
            {
                color = cImg.at<cv::Vec3b>(cur_y, cur_x);
                cov.at<float>(h, k) = cov.at<float>(h, k) + cur_w*color[h]*color[k];
            }
        }
        
        total_w += cur_w;
    }
    
    float inv_total_w = 1.0f/total_w;
    for(int h=0;h<3;h++)
    {
        for(int k=0;k<3;k++)
            cov.at<float>(h, k) = cov.at<float>(h, k) - inv_total_w*mean.at<float>(h, 0)*mean.at<float>(k, 0);
    }
}

void BayesianMatting::CalculateMeanCov(Mat cImg, vector<pair<cv::Point, float> > &clus_set, Mat mean, Mat cov)
{
    int cur_x, cur_y;
    float cur_w, total_w=0;
    Mat(Mat::zeros(mean.rows, mean.cols, CV_32FC1)).copyTo(mean);
    Mat(Mat::zeros(cov.rows, cov.cols, CV_32FC1)).copyTo(cov);    
    for(size_t j=0;j<clus_set.size();j++)
    {
        cur_x = clus_set[j].first.x;
        cur_y = clus_set[j].first.y;
        cur_w = clus_set[j].second;
        
        for(int h=0;h<3;h++)
        {
            cv::Vec3b color = cImg.at<cv::Vec3b>(cur_y, cur_x);
            mean.at<float>(h, 0) = mean.at<float>(h, 0) + cur_w*color[h];
            for(int k=0;k<3;k++)
            {
                color = cImg.at<cv::Vec3b>(cur_y, cur_x);
                cov.at<float>(h, k) = cov.at<float>(h, k) + cur_w*color[h]*color[k];
            }
        }
        
        total_w += cur_w;
    }
    
    float inv_total_w = 1.0f/total_w;
    for(int h=0;h<3;h++)
    {
        mean.at<float>(h, 0) = mean.at<float>(h, 0) * inv_total_w;
        for(int k=0;k<3;k++)
            cov.at<float>(h, k) = cov.at<float>(h, k) * inv_total_w;
    }
    
    for(int h=0;h<3;h++)
    {
        for(int k=0;k<3;k++)
        {
             cov.at<float>(h, k) = cov.at<float>(h, k) - mean.at<float>(h, 0)*mean.at<float>(k, 0);
        }
    }
}

void BayesianMatting::CalculateWeightMeanCov(Mat cImg, vector<pair<cv::Point, float> > &clus_set, float &weight, Mat mean, Mat cov)
{
    int cur_x, cur_y;
    float cur_w, total_w=0;
    Mat(Mat::zeros(mean.rows, mean.cols, CV_32FC1)).copyTo(mean);
    Mat(Mat::zeros(cov.rows, cov.cols, CV_32FC1)).copyTo(cov);    
    for(size_t j=0;j<clus_set.size();j++)
    {
        cur_x = clus_set[j].first.x;
        cur_y = clus_set[j].first.y;
        cur_w = clus_set[j].second;
        
        for(int h=0;h<3;h++)
        {
            cv::Vec3b color = cImg.at<cv::Vec3b>(cur_y, cur_x);
            mean.at<float>(h, 0) = mean.at<float>(h, 0) + cur_w*color[h];
            for(int k=0;k<3;k++)
            {
                //color = cImg.at<cv::Vec3b>(cur_y, cur_x);
                cov.at<float>(h, k) = cov.at<float>(h, k) + cur_w*color[h]*color[k];
            }
        }
        
        total_w += cur_w;
    }
    
    float inv_total_w = 1.0f/total_w;
    for(int h=0;h<3;h++)
    {
        mean.at<float>(h, 0) = mean.at<float>(h, 0) * inv_total_w;
        for(int k=0;k<3;k++)
            cov.at<float>(h, k) = cov.at<float>(h, k) * inv_total_w;
    }
    
    for(int h=0;h<3;h++)
    {
        for(int k=0;k<3;k++)
        {
             cov.at<float>(h, k) = cov.at<float>(h, k) - mean.at<float>(h, 0)*mean.at<float>(k, 0);
        }
    }
    
    weight = total_w;
}

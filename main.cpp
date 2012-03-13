#include <stdio.h>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Mat trimap;
bool bLeftButton = false;
bool bRightButton = false;

void onMouse( int event, int x, int y, int, void* param)
{
    if(event==CV_EVENT_LBUTTONUP)
    {
        bLeftButton = false;
    }
    
    if(event==CV_EVENT_RBUTTONUP)
    {
        bRightButton = false;
    }
    
    if(event==CV_EVENT_LBUTTONDOWN || bLeftButton)
    {
        // Foreground
        cv::circle(trimap, cv::Point(x,y), 10, CV_RGB(0,0,0), CV_FILLED);
        bLeftButton = true;
    }
    else if(event==CV_EVENT_RBUTTONDOWN || bRightButton)
    {
        // Background
        cv::circle(trimap, cv::Point(x,y), 10, CV_RGB(255,255,255), CV_FILLED);
        bRightButton = true;
    }
    else
    {
        bLeftButton = false;
        bRightButton = false;
    }
}

void clearPreviousSettings()
{
    trimap = Mat(img.size(), CV_8UC3, cv::Scalar(128,128,128));      // The actual trimap
    fgR = {0};
    fgG = {0};
    fgB = {0};
}

Mat getTrimap(Mat img)
{
    Mat imgDisplay = Mat(img.size(), CV_8UC3);  // The image displayed to the user
    
    cv::namedWindow("trimap");
    cv::setMouseCallback("trimap", onMouse, NULL);
    
    
    while(true)
    {
        imgDisplay = 0.5*trimap + img;
        cv::imshow("trimap", imgDisplay);
    
        // If the user presses 'Enter', return
        char input = cv::waitKey(15);
        
        if(input>=0)
            break;
    }
    
    return trimap;
}

int main(int argc, char** argv)
{
    // Get an input file as an argument
    if(argc<2)
    {
        // We don't have enough arguments
        printf("\nUsage:\n%s <filen1> <file2> <file3>\n\n", argv[0]);
        return 1;
    }
    
    for(int i=1;i<argc;i++)
    {
        // Load the image and try solving it
        Mat img = cv::imread(argv[i]);
        clearPreviousSettings();
        
        // Get a trimap from the user
        Mat imgTrimap = getTrimap(img);
        
        cv::waitKey(0);
    }

    return 0;
}

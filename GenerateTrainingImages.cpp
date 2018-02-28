#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


using namespace cv;
using namespace std;
int fcounter=0,ncounter=0;

int main( int argc, const char** argv)
{
  
  
  VideoCapture cam(0);
  if(!cam.isOpened()){
    cout<<"ERROR not opened "<< endl;
    return -1;
  }
  Mat img;
  Mat img_threshold;
  Mat img_gray;
  Mat img_roi;
  Mat img_res;
  //namedWindow("Original_image",CV_WINDOW_AUTOSIZE);//for the name of the windows
  //namedWindow("Gray_image",CV_WINDOW_AUTOSIZE);
  namedWindow("Thresholded_image",CV_WINDOW_AUTOSIZE);
  namedWindow("ROI",CV_WINDOW_AUTOSIZE);
  char a[40];
  int count =0;
  
  while(1){
    bool b=cam.read(img);//load image
    if(!b){
      cout<<"ERROR : cannot read"<<endl;
      return -1;
    }
    Rect roi(340,100,250,250); //defeining a rectangle for region of interest(roi) wih the given dimensions
    img_roi=img(roi); 
      cvtColor(img_roi,img_gray,CV_RGB2GRAY);

        GaussianBlur(img_gray,img_gray,Size(19,19),0.0,0);
        threshold(img_gray,img_threshold,0,255,THRESH_BINARY_INV+THRESH_OTSU);

   // skinSegment(img_roi, img_threshold);
  //  erode(img_threshold,img_threshold,Mat() );              
    //dilate(img_threshold, img_threshold, Mat() );
    
    cvtColor(img_roi,img_gray,CV_RGB2GRAY);//converts  image from one colorspace to other
    GaussianBlur(img_threshold,img_threshold,Size(19,19),0.0,0);//apply gaussian filter for reducing details not the fastest //test here others as bilateral
    threshold(img_threshold,img_threshold,0,255,THRESH_BINARY_INV+THRESH_OTSU);//for segmentation extracting what we want uses the otsu algorith to determine the optimal treshold
    fcounter++;
    std::cout<<fcounter;
    resize(img_threshold,img_res,Size(),0.08, 0.08, INTER_CUBIC);//creating an image with 20*20 
    if(fcounter>5){
      string fname="ti"+std::to_string(ncounter)+".jpg";
      imwrite(fname,img_res);
      fcounter=0;
      ncounter++;
    }
    //   imshow("Original_image",img);
    //     imshow("Gray_image",img_gray);
    imshow("Thresholded_image",img_threshold);
    imshow("ROI",img_roi);
    if(waitKey(30)==27){
      return -1;
    }
    
    
    
  }
  
  return 0;
}

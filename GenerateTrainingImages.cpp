#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


using namespace cv;
using namespace std;
int fcounter=0,ncounter=0;
void skinSegment2(const Mat& src, Mat& dst){
  dst.create( src.rows ,src.cols , CV_8U);
  for(int x=0;x<250;x++)
    {
      for(int y=0;y<250;y++)
      {
	//Vec4b intensity = img_roi.at<Vec4b>(y, x);
	Vec3b intensity=src.at<Vec3b>(y,x);
	//Vec3b int2=img_hsv.at<Vec3b>(y,x);
	
	int B = intensity.val[0];
	int G = intensity.val[1];
	int R = intensity.val[2];
	
	int a=(R<G)?G:R;
	a=(B<a)?a:B;
	// b=min(R,G,B);
	int b=(R>G)?G:R;
	b=(B>a)?a:B;
	int c=abs(R-G);
	//if((R>95)&&(G>40)&&(B>20)&&((a-b)>15)&&(c>15)&&(R>G)&&(R>B)){
	if((B>R)&&(G<B)&&((B<.25*R||B>200))){
	//cout<<"got Skin";
	 dst.at<uchar>(y,x)=0;
	}
	else{
	  	  dst.at<uchar>(y,x)=255;
	}
	
      }
    }
}
void skinSegment(const Mat& src, Mat& dst){
  dst.create( src.rows ,src.cols , CV_8U);
  Mat imgYCrCb;
  
  cvtColor(src, imgYCrCb, CV_BGR2YCrCb);
  
  vector<Mat> ycrcb( imgYCrCb.channels());
  split(imgYCrCb, ycrcb); 
  int y, cr, cb, x1, y1, value;
  int iRows=src.rows;
  int iCols=src.cols;
  for (int h=0; h<iRows; h++){              // h:行
    uchar* data_YCrCb0=ycrcb[0].ptr<uchar>(h);
    uchar* data_YCrCb1=ycrcb[1].ptr<uchar>(h);
    uchar* data_YCrCb2=ycrcb[2].ptr<uchar>(h);
    
    const uchar* d1 = src.ptr<uchar>(h);
    uchar* data_dst = dst.ptr<uchar>(h);
    for (int w=0; w<iCols ;w++){
      y = data_YCrCb0[w];
      cr = data_YCrCb1[w];
      cb = data_YCrCb2[w];
      cb -= 109;
      cr -= 152;
      x1 = (819*cr-614*cb)/32 + 51;
      y1 = (819*cr+614*cb)/32 + 77;
      x1 = x1*41/1024;
      y1 = y1*73/1024;
      value = x1*x1+y1*y1;
      if(y<100)	
	data_dst[w] =(value<500) ? 255:0;
      else	
	data_dst[w] =(value<750)? 255:0;
    }
    
  }
}
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
    Rect roi(340,100,250,250); //defeining a rectangle called roi wih the given dimensions
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

#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;
int fcounter=0,ncounter=0;
void readCsv(vector<vector<double>> &fields,string fname){
  ifstream in(fname);
  
  if (in) {
    string line;
    while (getline(in, line)) {
      stringstream sep(line);
      string field;
      fields.push_back(vector<double>());
      while (getline(sep, field, ',')) {
	fields.back().push_back(stod(field));
      }
    }
  }
}

void setVector(vector<vector<double> >& vec, int x_dim, int y_dim){
  vec.resize(x_dim);
  for(int i=0; i < vec.size(); i++)
    vec[i].resize(y_dim);
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
  std::vector<std::vector<double>>Theta1;
  std::vector<std::vector<double>>Theta2;
  readCsv(Theta1,"Theta1.csv");
  readCsv(Theta2,"Theta2.csv");
  
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
    //fcounter++;
    //std::cout<<fcounter;
    resize(img_threshold,img_res,Size(),0.08, 0.08, INTER_CUBIC);//creating an image with 20*20 
    /* if(fcounter>5){
     *      string fname="ti"+std::to_string(ncounter)+".jpg";
     *      imwrite(fname,img_res);
     *      fcounter=0;
     *      ncounter++;
  }*/
    
    std::vector<double>h1;
    std::vector<double>h2;
    std::vector<int>input;
    input.resize(401);
    h1.resize(50);
    h2.resize(4);
    //setVector(input,401,1);
    input[0]=1;
    int cc=1;
    for(int i=0;i<img_res.rows;i++){
      for(int j=0;j<img_res.cols;j++){
	input[cc]=int(img_res.at<uchar>(i, j));
	cc++;
      }
    }
    
    
    int x1,x2,x;
    setVector(Theta1,50,401);
    setVector(Theta2,4,51);
    //Forward Propagation
    h1[0]=1;
    
    for(int j=0;j<50;j++)
    {
      
      double z= 0;
      
      for(int k=0;k<401;k++)
      {
	z +=input[k] * Theta1[j][k];
      }
      h1[j+1]=1.0 / (1.0 + exp(-z));
    }
    
    for(int j=0;j<4;j++)
    {
      
      double z = 0;
      
      for(int k=0;k<51;k++)
      {
	z +=h1[k] * Theta2[j][k];
      }
      h2[j]=1.0 / (1.0 + exp(-z));
    }
    cout<<h2[0]<<","<<h2[1]<<","<<h2[2]<<","<<h2[3]<<endl;
    //findin max i thought its fast may be use more fast algorithm here
    double max1,max2;
    if(h2[0]>h2[1]){
      x1=0;
      max1=h2[0];
    }
    else{
      x1=1;
    max1=h2[1];
    }
    if(h2[2]>h2[3]){
      x2=2;
    max2=h2[2];
    }
    else{
      x2=3;
      max2=h2[3];
    }
    if(max2>max1)
      x=x2;
    else
      x=x1;
    if(x==0)
      strcpy(a,"Wave ");
    else if(x==1)
      strcpy(a,"Number One ");
    else if(x==2)
      strcpy(a,"Victory");
    else
      strcpy(a,"Welcome !!");
    
    putText(img,a,Point(70,70),CV_FONT_HERSHEY_SIMPLEX,3,Scalar(255,0,0),2,8,false);
    imshow("Original_image",img);
    //     imshow("Gray_image",img_gray);
    imshow("Thresholded_image",img_threshold);
    imshow("ROI",img_roi);
    if(waitKey(30)==27){
      return -1;
    }
    
    
    
  }
  
  return 0;
}

#include <string>
#include <cstdlib>
#include <vector>
#include <utility>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "LBPFeatureExtractor.h"

using namespace std;
using namespace cv;

LBPFeatureExtractor::LBPFeatureExtractor(const vector<int> scale, const int patchSize, const int numCellX, const int numCellY, const bool uniform) {
   setParameters(scale, patchSize, numCellX, numCellY, uniform);
   int _array[256] = {0, 1, 2, 3, 4, 58, 5, 6, 7, 58, 58, 58, 8, 58, 9, 10, 11, 58, 58, 58, 58, 58, 58, 58, 12, 58, 58, 58, 13, 58, 14, 15, 16, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 17, 58, 58, 58, 58, 58, 58, 58, 18, 58, 58, 58, 19, 58, 20, 21, 22, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 23, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 24, 58, 58, 58, 58, 58, 58, 58, 25, 58, 58, 58, 26, 58, 27, 28, 29, 30, 58, 31, 58, 58, 58, 32, 58, 58, 58, 58, 58, 58, 58, 33, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 34, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 35, 36, 37, 58, 38, 58, 58, 58, 39, 58, 58, 58, 58, 58, 58, 58, 40, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 41, 42, 43, 58, 44, 58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46, 47, 48, 58, 49, 58, 58, 58, 50, 51, 52, 58, 53, 54, 55, 56, 57};
   memcpy(mapping,_array,256 * sizeof(int));
}
void LBPFeatureExtractor::setParameters(const vector<int> scale, const int patchSize, const int numCellX, const int numCellY, const bool uniform){
   for(int i=0; i<scale.size(); i++) {
      this->scale.push_back(scale[i]);
   }
   this->patchSize = patchSize;
   this->numCellX = numCellX;
   this->numCellY = numCellY;
   this->uniform = uniform;
}
int* LBPFeatureExtractor::extractAt(const Mat &inputImage, const vector< pair<double, double> >points, int *outputFeature){
   
   Mat processedImage;
   Mat resizeImage;
   //convert image into graylevel
   if (inputImage.channels() == 3) {
      cvtColor(inputImage, processedImage, CV_BGR2GRAY );
   } else {
      processedImage = inputImage;
   }
   int count = 0;
   for(int s = 0; s<this->scale.size(); s++)
   {
      int currentScale = this->scale[s];
      //scale face into proper size
      vector< pair<double, double> > newPoints;
      for(int i=0; i<points.size(); i++)
      {
         pair<double,double> point;
         point.first = points[i].first*double(currentScale)/inputImage.cols;
         point.second = points[i].second*double(currentScale)/inputImage.rows;
         newPoints.push_back(point);
      }
      
      //printf("%d %d %d\n", currentScale, inputImage.cols, inputImage.rows);
      equalizeHist(processedImage, processedImage);
      resize(processedImage, resizeImage, Size(currentScale, currentScale));
      //compute center and extract lbp feature
      Mat patch;
      int hist[59];
      for(int i=0; i<newPoints.size(); i++)
      {
         for(int j=0; j<numCellX; j++)
            for(int k=0; k<numCellY; k++)
            {
               double centerX = int(newPoints[i].first - patchSize*(numCellX/2) + (numCellX%2 == 0)*patchSize*0.5 + patchSize*j);
               double centerY = int(newPoints[i].second - patchSize*(numCellY/2) + (numCellY%2 == 0)*patchSize*0.5 + patchSize*k);
               //circle(resizeImage, Point(centerX,centerY), 5, Scalar(255,0,0), CV_FILLED);
               getRectSubPix(resizeImage, Size(patchSize+2, patchSize+2), Point2f(centerX,centerY), patch);
               lbpHist(patch, hist);
               for(int l=0; l<59; l++)
                  outputFeature[count++] = hist[l];
            }
      }
      //imwrite(to_string(currentScale)+"out.jpg", resizeImage);
      resizeImage.release();
   }
   processedImage.release();

}

void LBPFeatureExtractor::lbpImage(const Mat &img, Mat &outImage){
   unsigned char locP[9];
   for(int i = 1; i<img.rows-1; i++)
      for(int j = 1; j< img.cols-1; j++)
      {
         locP[0] = img.at<unsigned char>(i,j);
         locP[1] = img.at<unsigned char>(i-1,j);
         locP[2] = img.at<unsigned char>(i-1,j-1);
         locP[3] = img.at<unsigned char>(i,j-1);
         locP[4] = img.at<unsigned char>(i+1,j-1);
         locP[5] = img.at<unsigned char>(i+1,j);
         locP[6] = img.at<unsigned char>(i+1,j+1);
         locP[7] = img.at<unsigned char>(i,j+1);
         locP[8] = img.at<unsigned char>(i-1,j+1);
         outImage.at<unsigned char>(i, j) = lbpCode(locP);
      }
}

//Compute an single lbp code from a pixel
int LBPFeatureExtractor::lbpCode (unsigned char seq[9])
{
   bool lab[8] = {false};
   int base = seq[0];
   int result = 0, one = 1, final;
   int i;
   for (i = 0; i < 8; i++) {
      if (base >= seq[i+1])
         lab[i] = 1;
      else
         lab[i] = 0;
   }

   for (i = 0; i < 8; i++, one = one << 1)
      result += lab[i]*one;
   final = mapping[result];
   return final;
}

//Compute lbp histgoram from an image
int* LBPFeatureExtractor::lbpHist(Mat &img, int* lbpHist)
{
   unsigned char locP[9];
   for(int i=0; i<59; i++)
      lbpHist[i] = 0;
   for(int i = 1; i<img.rows-1; i++)
      for(int j = 1; j< img.cols-1; j++)
      {  
         locP[0] = img.at<unsigned char>(i,j);
         locP[1] = img.at<unsigned char>(i-1,j);
         locP[2] = img.at<unsigned char>(i-1,j-1);
         locP[3] = img.at<unsigned char>(i,j-1);
         locP[4] = img.at<unsigned char>(i+1,j-1);
         locP[5] = img.at<unsigned char>(i+1,j);
         locP[6] = img.at<unsigned char>(i+1,j+1);
         locP[7] = img.at<unsigned char>(i,j+1);
         locP[8] = img.at<unsigned char>(i-1,j+1);
         lbpHist[lbpCode(locP)]++;
      }

   return lbpHist;
}


#include <string>
#include <vector>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

class FaceDetector{
public:
   FaceDetector(const string& modelFileName);
   bool setDetector(const string& modelFileName);
   vector<Rect>& detect(const Mat& inputImage, vector<Rect> &outputFaces, Size minFaceSize=Size(30,30), float scaleFactors=1.1, int minNeighbors=4);
   //bool detectAndSave(const Mat& inputImage, const string& outputFileName);
private:
   CascadeClassifier cascade;
};

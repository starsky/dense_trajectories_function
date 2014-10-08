#ifndef DENSETRACK_H_
#define DENSETRACK_H_

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <ctype.h>
#include <unistd.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <string>

using namespace cv;
/*
int start_frame = 0;
int end_frame = INT_MAX;
int scale_num = 8;
const float scale_stride = sqrt(2);

// parameters for descriptors
int patch_size = 32;
int nxy_cell = 2;
int nt_cell = 3;
float epsilon = 0.05;
const float min_flow = 0.4;

// parameters for tracking
double quality = 0.001;
int min_distance = 5;
int init_gap = 1;
int track_length = 15;
*/


typedef struct {
	int x;       // top left corner
	int y;
	int width;
	int height;
}RectInfo;

typedef struct {
    int width;   // resolution of the video
    int height;
    int length;  // number of frames
}SeqInfo;

typedef struct {
    int length;  // length of the trajectory
    int gap;     // initialization gap for feature re-sampling 
}TrackInfo;

typedef struct {
    int nBins;   // number of bins for vector quantization
    bool isHof; 
    int nxCells; // number of cells in x direction
    int nyCells; 
    int ntCells;
    int dim;     // dimension of the descriptor
    int height;  // size of the block for computing the descriptor
    int width;
}DescInfo; 

// integral histogram for the descriptors
typedef struct {
    int height;
    int width;
    int nBins;
    float* desc;
}DescMat;

class Track
{
public:
    std::vector<Point2f> point;
    std::vector<float> hog;
    std::vector<float> hof;
    std::vector<float> mbhX;
    std::vector<float> mbhY;
    int index;

    Track(const Point2f& point_, const TrackInfo& trackInfo, const DescInfo& hogInfo,
          const DescInfo& hofInfo, const DescInfo& mbhInfo)
        : point(trackInfo.length+1), hog(hogInfo.dim*trackInfo.length),
          hof(hofInfo.dim*trackInfo.length), mbhX(mbhInfo.dim*trackInfo.length), mbhY(mbhInfo.dim*trackInfo.length)
    {
        index = 0;
        point[0] = point_;
    }

    void addPoint(const Point2f& point_)
    {
        index++;
        point[index] = point_;
    }
};

class DenseTrajectories {
	private:
		// parameters for rejecting trajectory
		const float min_var = sqrt(3);
		const float max_var = 50;
		const float max_dis = 20;
		const float scale_stride = sqrt(2);
		const float min_flow = 0.4;

		int start_frame = 0;
		int end_frame = INT_MAX;
		int scale_num = 8;

		// parameters for descriptors
		int patch_size = 32;
		int nxy_cell = 2;
		int nt_cell = 3;
		float epsilon = 0.05;
		 // parameters for tracking
		double quality = 0.001;
		int min_distance = 5;
		int init_gap = 1;
		int track_length = 15;
		int show_track = 0; // set show_track = 1, if you want to visualize the trajectories

		Mat image, prev_grey, grey;
		std::vector<float> fscales{std::vector<float>(0)};
		std::vector<Size> sizes{std::vector<Size>(0)};
		std::vector<Mat> prev_grey_pyr{std::vector<Mat>(0)}, grey_pyr{std::vector<Mat>(0)}, flow_pyr{std::vector<Mat>(0)};
		std::vector<Mat> prev_poly_pyr{std::vector<Mat>(0)}, poly_pyr{std::vector<Mat>(0)}; // for optical flow
		std::vector<std::list<Track> > xyScaleTracks;

		int init_counter = 0; // indicate when to detect new feature points
		int frame_num = 0;
		TrackInfo trackInfo;
		DescInfo hogInfo, hofInfo, mbhInfo;
		
		bool export_stats = true;
		bool export_tracklets = true;
		bool export_hog = true;
		bool export_hof = true;
		bool export_mbhx = false;
		bool export_mbhy = false;
		bool export_mbh_whole = true;	
	public:
		void initialize_dense_track();
		void process_frame(Mat& frame, std::vector<cv::Mat >* results);
		bool arg_parse(int argc, char** argv);
		void printVect(std::vector< std::vector< float > >& featuresVect);
		void printMat(std::vector<cv::Mat >& featuresVect);
		void GetRect(const Point2f& point, RectInfo& rect, const int width, const int height, const DescInfo& descInfo);
		void BuildDescMat(const Mat& xComp, const Mat& yComp, float* desc, const DescInfo& descInfo);
		void GetDesc(const DescMat* descMat, RectInfo& rect, DescInfo descInfo, std::vector<float>& desc, const int index);
		void HogComp(const Mat& img, float* desc, DescInfo& descInfo);
		void HofComp(const Mat& flow, float* desc, DescInfo& descInfo);
		void MbhComp(const Mat& flow, float* descX, float* descY, DescInfo& descInfo);
		void DenseSample(const Mat& grey, std::vector<Point2f>& points, const double quality, const int min_distance);
		void InitPry(const Mat& frame, std::vector<float>& scales, std::vector<Size>& sizes);
		void BuildPry(const std::vector<Size>& sizes, const int type, std::vector<Mat>& grey_pyr);
		void DrawTrack(const std::vector<Point2f>& point, const int index, const float scale, Mat& image);
		int AppendVectDesc(std::vector<float>& desc, DescInfo& descInfo, TrackInfo& trackInfo, cv::Mat& row, int start_column);
		bool IsValid(std::vector<Point2f>& track, float& mean_x, float& mean_y, float& var_x, float& var_y, float& length);
};

#endif /*DENSETRACK_H_*/

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

class DenseTrajectoriesBuilder;

/**
 * Class for computation of dense trajectories features.
 *
 * This class contains all code for computing dense trajectories features.
 * It has been adopted from: http://lear.inrialpes.fr/people/wang/dense_trajectories
 * and put into object oriented way.
 */ 
class DenseTrajectories {
	friend class DenseTrajectoriesBuilder;
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
	private:
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
	private:
		DenseTrajectories(int in_start_frame, int in_end_frame, int in_track_length, int in_min_distance,
			int in_patch_size, int in_nxy_cell, int in_nt_cell, int in_scale_num, int in_init_gap,
			bool in_export_header, bool in_export_trajectories, bool in_export_hog, bool in_export_hof,
			bool in_export_mbhx, bool in_export_mbhy, bool in_export_mbh) : 
			start_frame(in_start_frame), end_frame(in_end_frame), track_length(in_track_length),
			min_distance(in_min_distance), patch_size(in_patch_size), nxy_cell(in_nxy_cell),
			nt_cell(in_nt_cell), scale_num(in_scale_num), init_gap(in_init_gap),
			export_stats(in_export_header),	export_tracklets(in_export_trajectories),
			export_hog(in_export_hog), export_hof(in_export_hof), export_mbhx(in_export_mbhx),
			export_mbhy(in_export_mbhy), export_mbh_whole(in_export_mbh) {
		
			initialize_dense_track();
		}
		void initialize_dense_track();
	public:
		/**
         * Process input frame, and computes dense trajectories features.
         *
         * Computes dense trajectories features based on input parameter frame. The features computed based
         * on previous and current frame are saved in results output parameter.
         * Output vector has as many elements as features selected to export using set_export_* in
         * DenseTrajectoriesBuilder. Each element of vector contains cv::Mat object with computed features,
         * based on current and previous frames.
         */
		void process_frame(const Mat& frame, std::vector<cv::Mat >& results);
		/**
		 * Prints computed features to stdout.
		 */ 
		void printMat(const std::vector<cv::Mat >& featuresVect) const;
	};
/**
 * Builder class for creating DenseTrajectories object.
 *
 * Before invoking create() method it is possible
 * to change parameters of feature extraction. To check actual meanings of given parameters please
 * go to the website: http://lear.inrialpes.fr/people/wang/dense_trajectories
 *
 * set_export_* control which descriptors are included in resulting feature vector.
 */
class DenseTrajectoriesBuilder {
	private:
		int start_frame = 0;
		int end_frame = INT_MAX;
		int scale_num = 8;
		// parameters for descriptors
		int patch_size = 32;
		int nxy_cell = 2;
		int nt_cell = 3;
		//parameters for tracking
		int min_distance = 5;
		int init_gap = 1;
		int track_length = 15;
		//export controll options
		bool export_stats = true;
		bool export_tracklets = true;
		bool export_hog = true;
		bool export_hof = true;
		bool export_mbhx = false;
		bool export_mbhy = false;
		bool export_mbh_whole = true;	
	public:
		void set_start_frame(int start_frame);
		void set_end_frame(int end_frame);
		void set_track_length(int track_length);
		void set_min_distance(int min_distance);
		void set_patch_size(int patch_size);
		void set_nxy_cell(int nxy_cell);
		void set_nt_cell(int nt_cell);
		void set_scale_num(int scale_num);
		void set_init_gap(int init_gap);
		void set_export_header(bool use_header);
		void set_export_trajectories(bool export_trajectories);
		void set_export_hog(bool export_hog);
		void set_export_hof(bool export_hof);
		void set_export_mbhx(bool export_mbhx);
		void set_export_mbhy(bool export_mbhy);
		void set_export_mbh(bool export_mbh);
		/**
		 * Creates ready to use DenseTrajectories object
		 */
		DenseTrajectories& create();
};
#endif /*DENSETRACK_H_*/

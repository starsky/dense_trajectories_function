#include "DenseTrackLib.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"

#include <time.h>

using namespace cv;

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

//Global variables (soo ugly :( )
std::vector<float> fscales(0);
std::vector<Size> sizes(0);

std::vector<Mat> prev_grey_pyr(0), grey_pyr(0), flow_pyr(0);
std::vector<Mat> prev_poly_pyr(0), poly_pyr(0); // for optical flow

std::vector<std::list<Track> > xyScaleTracks;
int init_counter = 0; // indicate when to detect new feature points
int frame_num = 0;
TrackInfo trackInfo;
DescInfo hogInfo, hofInfo, mbhInfo;


void initialize_dense_track() {
	InitTrackInfo(&trackInfo, track_length, init_gap);
	InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&hofInfo, 9, true, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&mbhInfo, 8, false, patch_size, nxy_cell, nt_cell);


	if(show_track == 1)
		namedWindow("DenseTrack", 0);
}

void process_frame(Mat& frame, std::vector< std::vector< float > >* results) {
	int i, j, c;
	if(frame.empty())
		return;

	if(frame_num < start_frame || frame_num > end_frame) {
		frame_num++;
		return;
	}

	if(frame_num == start_frame) {
		image.create(frame.size(), CV_8UC3);
		grey.create(frame.size(), CV_8UC1);
		prev_grey.create(frame.size(), CV_8UC1);

		InitPry(frame, fscales, sizes);

		BuildPry(sizes, CV_8UC1, prev_grey_pyr);
		BuildPry(sizes, CV_8UC1, grey_pyr);

		BuildPry(sizes, CV_32FC2, flow_pyr);
		BuildPry(sizes, CV_32FC(5), prev_poly_pyr);
		BuildPry(sizes, CV_32FC(5), poly_pyr);

		xyScaleTracks.resize(scale_num);

		frame.copyTo(image);
		cvtColor(image, prev_grey, CV_BGR2GRAY);

		for(int iScale = 0; iScale < scale_num; iScale++) {
			if(iScale == 0)
				prev_grey.copyTo(prev_grey_pyr[0]);
			else
				resize(prev_grey_pyr[iScale-1], prev_grey_pyr[iScale], prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

			// dense sampling feature points
			std::vector<Point2f> points(0);
			DenseSample(prev_grey_pyr[iScale], points, quality, min_distance);

			// save the feature points
			std::list<Track>& tracks = xyScaleTracks[iScale];
			for(i = 0; i < points.size(); i++)
				tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
		}

		// compute polynomial expansion
		my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr, fscales, 7, 1.5);

		frame_num++;
		return;
	}

	init_counter++;
	frame.copyTo(image);
	cvtColor(image, grey, CV_BGR2GRAY);

	// compute optical flow for all scales once
	my::FarnebackPolyExpPyr(grey, poly_pyr, fscales, 7, 1.5);
	my::calcOpticalFlowFarneback(prev_poly_pyr, poly_pyr, flow_pyr, 10, 2);

	for(int iScale = 0; iScale < scale_num; iScale++) {
		if(iScale == 0)
			grey.copyTo(grey_pyr[0]);
		else
			resize(grey_pyr[iScale-1], grey_pyr[iScale], grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

		int width = grey_pyr[iScale].cols;
		int height = grey_pyr[iScale].rows;

		// compute the integral histograms
		DescMat* hogMat = InitDescMat(height+1, width+1, hogInfo.nBins);
		HogComp(prev_grey_pyr[iScale], hogMat->desc, hogInfo);

		DescMat* hofMat = InitDescMat(height+1, width+1, hofInfo.nBins);
		HofComp(flow_pyr[iScale], hofMat->desc, hofInfo);

		DescMat* mbhMatX = InitDescMat(height+1, width+1, mbhInfo.nBins);
		DescMat* mbhMatY = InitDescMat(height+1, width+1, mbhInfo.nBins);
		MbhComp(flow_pyr[iScale], mbhMatX->desc, mbhMatY->desc, mbhInfo);

		// track feature points in each scale separately
		std::list<Track>& tracks = xyScaleTracks[iScale];
		for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();) {
			int index = iTrack->index;
			Point2f prev_point = iTrack->point[index];
			int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width-1);
			int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height-1);

			Point2f point;
			point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];
			point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1];

			if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
				iTrack = tracks.erase(iTrack);
				continue;
			}

			// get the descriptors for the feature point
			RectInfo rect;
			GetRect(prev_point, rect, width, height, hogInfo);
			GetDesc(hogMat, rect, hogInfo, iTrack->hog, index);
			GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
			GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
			GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
			iTrack->addPoint(point);

			// draw the trajectories at the first scale
			if(show_track == 1 && iScale == 0)
				DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image);

			// if the trajectory achieves the maximal length
			if(iTrack->index >= trackInfo.length) {
				std::vector<Point2f> trajectory(trackInfo.length+1);
				for(int i = 0; i <= trackInfo.length; ++i)
					trajectory[i] = iTrack->point[i]*fscales[iScale];
			
				float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
				if(IsValid(trajectory, mean_x, mean_y, var_x, var_y, length)) {
					std::vector<float> row;
					//printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t", frame_num, mean_x, mean_y, var_x, var_y, length, fscales[iScale]);
					row.push_back(frame_num);
					row.push_back(mean_x);
					row.push_back(mean_y);
					row.push_back(var_x);
					row.push_back(var_y);
					row.push_back(length);
					row.push_back(fscales[iScale]);
					// output the trajectory
					for (int i = 0; i < trackInfo.length; ++i) {
						row.push_back(trajectory[i].x);
						row.push_back(trajectory[i].y);
					}
					//	printf("%f\t%f\t", trajectory[i].x,trajectory[i].y);
					
					AppendVectDesc(iTrack->hog, hogInfo, trackInfo, row);
					AppendVectDesc(iTrack->hof, hofInfo, trackInfo, row);
					AppendVectDesc(iTrack->mbhX, mbhInfo, trackInfo, row);
					AppendVectDesc(iTrack->mbhY, mbhInfo, trackInfo, row);
					//PrintDesc(iTrack->hog, hogInfo, trackInfo);
					//PrintDesc(iTrack->hof, hofInfo, trackInfo);
					//PrintDesc(iTrack->mbhX, mbhInfo, trackInfo);
					//PrintDesc(iTrack->mbhY, mbhInfo, trackInfo);
					//printf("\n");
					results->push_back(row);
				}

				iTrack = tracks.erase(iTrack);
				continue;
			}
			++iTrack;
		}
		ReleDescMat(hogMat);
		ReleDescMat(hofMat);
		ReleDescMat(mbhMatX);
		ReleDescMat(mbhMatY);

		if(init_counter != trackInfo.gap)
			continue;

		// detect new feature points every initGap frames
		std::vector<Point2f> points(0);
		for(std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++)
			points.push_back(iTrack->point[iTrack->index]);

		DenseSample(grey_pyr[iScale], points, quality, min_distance);
		// save the new feature points
		for(i = 0; i < points.size(); i++)
			tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
	}

	init_counter = 0;
	grey.copyTo(prev_grey);
	for(i = 0; i < scale_num; i++) {
		grey_pyr[i].copyTo(prev_grey_pyr[i]);
		poly_pyr[i].copyTo(prev_poly_pyr[i]);
	}

	frame_num++;

	if( show_track == 1 ) {
		imshow( "DenseTrack", image);
		c = cvWaitKey(3);
		if((char)c == 27) return;
	}
	return;
}

void usage()
{
	fprintf(stderr, "Extract dense trajectories from a video\n\n");
	fprintf(stderr, "Usage: DenseTrack video_file [options]\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "  -h                        Display this message and exit\n");
	fprintf(stderr, "  -S [start frame]          The start frame to compute feature (default: S=0 frame)\n");
	fprintf(stderr, "  -E [end frame]            The end frame for feature computing (default: E=last frame)\n");
	fprintf(stderr, "  -L [trajectory length]    The length of the trajectory (default: L=15 frames)\n");
	fprintf(stderr, "  -W [sampling stride]      The stride for dense sampling feature points (default: W=5 pixels)\n");
	fprintf(stderr, "  -N [neighborhood size]    The neighborhood size for computing the descriptor (default: N=32 pixels)\n");
	fprintf(stderr, "  -s [spatial cells]        The number of cells in the nxy axis (default: nxy=2 cells)\n");
	fprintf(stderr, "  -t [temporal cells]       The number of cells in the nt axis (default: nt=3 cells)\n");
	fprintf(stderr, "  -A [scale number]         The number of maximal spatial scales (default: 8 scales)\n");
	fprintf(stderr, "  -I [initial gap]          The gap for re-sampling feature points (default: 1 frame)\n");
}

bool arg_parse(int argc, char** argv)
{
	int c;
	bool flag = false;
	char* executable = basename(argv[0]);
	while((c = getopt (argc, argv, "hS:E:L:W:N:s:t:A:I:")) != -1)
	switch(c) {
		case 'S':
		start_frame = atoi(optarg);
		flag = true;
		break;
		case 'E':
		end_frame = atoi(optarg);
		flag = true;
		break;
		case 'L':
		track_length = atoi(optarg);
		break;
		case 'W':
		min_distance = atoi(optarg);
		break;
		case 'N':
		patch_size = atoi(optarg);
		break;
		case 's':
		nxy_cell = atoi(optarg);
		break;
		case 't':
		nt_cell = atoi(optarg);
		break;
		case 'A':
		scale_num = atoi(optarg);
		break;
		case 'I':
		init_gap = atoi(optarg);
		break;	

		case 'h':
		usage();
		exit(0);
		break;

		default:
		fprintf(stderr, "error parsing arguments at -%c\n  Try '%s -h' for help.", c, executable );
		abort();
	}
	return flag;
}

void printVect(std::vector< std::vector< float > >& featuresVect) {
	int j = 0;
	for( std::vector< std::vector<float> >::const_iterator i = featuresVect.begin(); i != featuresVect.end(); ++i) {
		std::vector<float> r = *i;
		printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t", (int) r[0], r[1], r[2], r[3], r[4], r[5], r[6]);          
		j = 7;
		for(int z = 0; z < 15; z++) {
			printf("%f\t%f\t", r[j], r[j+1]);
			j += 2;
		}
		for(int z = j; z < r.size(); z++) {
			printf("%.7f\t", r[z]);
		}
		printf("\n");
	}
}
/*
int main(int argc, char** argv)
{
	VideoCapture capture;
	char* video = argv[1];
	int flag = arg_parse(argc, argv);
	capture.open(video);

	if(!capture.isOpened()) {
		fprintf(stderr, "Could not initialize capturing..\n");
		return -1;
	}
	initialize_dense_track();
	while(true) {
		// get a new frame
		Mat frame;
		capture >> frame;
		if(frame.empty())
			break;
		process_frame(frame);		
	}

	if( show_track == 1 )
		destroyWindow("DenseTrack");

	return 0;
}*/

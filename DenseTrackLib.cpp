#include "DenseTrackLib.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"

#include <time.h>

using namespace cv;


void DenseTrajectories::initialize_dense_track() {
	InitTrackInfo(&trackInfo, track_length, init_gap);
	//printf("init: %d %d\n", nxy_cell,nt_cell);
	InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);
	//printf("init hog %d\n", hogInfo.dim);
	InitDescInfo(&hofInfo, 9, true, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&mbhInfo, 8, false, patch_size, nxy_cell, nt_cell);


	if(show_track == 1)
		namedWindow("DenseTrack", 0);
}

void DenseTrajectories::process_frame(Mat& frame, std::vector<cv::Mat >* results) {
	if(export_stats) {
		cv::Mat row(0,7,CV_32F);
		results->push_back(row);
	}
	if(export_tracklets) {
		cv::Mat row(0,trackInfo.length * 2,CV_32F);
		results->push_back(row);
	}
	if(export_hog) {
		cv::Mat row(0,hogInfo.dim,CV_32F);
		results->push_back(row);
	}
	if(export_hof) {
		cv::Mat row(0,hofInfo.dim,CV_32F);
		results->push_back(row);
	}
	if(export_mbhx) {
		cv::Mat row(0,mbhInfo.dim,CV_32F);
		results->push_back(row);
	}
	if(export_mbhy) {
		cv::Mat row(0,mbhInfo.dim,CV_32F);
		results->push_back(row);
	}
	if(export_mbh_whole) {
		cv::Mat row(0,2 * mbhInfo.dim,CV_32F);
		results->push_back(row);
	}

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
	my::calcOpticalFlowFarneback(prev_poly_pyr, poly_pyr, flow_pyr, 10, 2, scale_stride);

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
					int curr_desc = 0;
					if(export_stats) {
						cv::Mat row(1,7,CV_32F);
						row.at<int>(0,0) =  frame_num;
						row.at<float>(0,1) = (mean_x);
						row.at<float>(0,2) = (mean_y);
						row.at<float>(0,3) = (var_x);
						row.at<float>(0,4) = (var_y);
						row.at<float>(0,5) = (length);
						row.at<float>(0,6) = (fscales[iScale]);
						results->at(curr_desc++).push_back(row);
					}
					// output the trajectory
					if(export_tracklets) {
						cv::Mat row(1,trackInfo.length * 2,CV_32F);
						int curr_col = 0;
						for (int i = 0; i < trackInfo.length; ++i) {
							row.at<float>(0,curr_col++) = (trajectory[i].x);
							row.at<float>(0,curr_col++) = (trajectory[i].y);
						}
						results->at(curr_desc++).push_back(row);
					}
					if(export_hog) {
						cv::Mat row(1,hogInfo.dim * hogInfo.ntCells, CV_32F);
						AppendVectDesc(iTrack->hog, hogInfo, trackInfo, row, 0);
						results->at(curr_desc++).push_back(row);
					}
					if(export_hof) {
						cv::Mat row(1,hofInfo.dim * hofInfo.ntCells, CV_32F);
						AppendVectDesc(iTrack->hof, hofInfo, trackInfo, row, 0);
						results->at(curr_desc++).push_back(row);
					}

					if(export_mbhx) {
						cv::Mat row(1,mbhInfo.dim * mbhInfo.ntCells, CV_32F);
						AppendVectDesc(iTrack->mbhX, mbhInfo, trackInfo, row, 0);
						results->at(curr_desc++).push_back(row);
					}

					if(export_mbhy) {
						cv::Mat row(1,mbhInfo.dim * mbhInfo.ntCells, CV_32F);
						AppendVectDesc(iTrack->mbhY, mbhInfo, trackInfo, row, 0);
						results->at(curr_desc++).push_back(row);
					}
					if(export_mbh_whole) {
						cv::Mat row(1,2 * mbhInfo.dim * mbhInfo.ntCells, CV_32F);
                                                int start_col = AppendVectDesc(iTrack->mbhX, mbhInfo, trackInfo, row, 0);
						AppendVectDesc(iTrack->mbhY, mbhInfo, trackInfo, row, start_col);
                                                results->at(curr_desc++).push_back(row);
					}

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



void DenseTrajectories::printMat(std::vector<cv::Mat >& vect) {
	int j = 0;
	int rows_count = vect.at(0).rows;
	for(int a = 0; a < rows_count; a++) {
		int curr_desc = 0;

		cv::Mat r = vect.at(curr_desc++);
		printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t", r.at<int>(a, 0), r.at<float>(a,1), r.at<float>(a,2), r.at<float>(a,3), r.at<float>(a,4), r.at<float>(a,5), r.at<float>(a,6));          
		j = 0;
		r = vect.at(curr_desc++);
		for(int z = 0; z < 15; z++) {
			printf("%f\t%f\t", r.at<float>(a,j), r.at<float>(a,j+1));
			j += 2;
		}

		for(;curr_desc < 5; curr_desc++) {
			cv::Mat r = vect.at(curr_desc);
			for(int z = 0; z < r.cols; z++) {
				printf("%.7f\t", r.at<float>(a,z));
			}
		}

		printf("\n");
	}
}

void DenseTrajectories::set_start_frame(int start_frame) {
	this->start_frame = start_frame;
}

void DenseTrajectories::set_end_frame(int end_frame) {
	this->end_frame = end_frame;
}

void DenseTrajectories::set_track_length(int track_length) {
	this->track_length = track_length;
}

void DenseTrajectories::set_min_distance(int min_distance) {
	this->min_distance = min_distance;
}

void DenseTrajectories::set_patch_size(int patch_size) {
	this->patch_size = patch_size;
}

void DenseTrajectories::set_nxy_cell(int nxy_cell) {
	this->nxy_cell = nxy_cell;
}

void DenseTrajectories::set_nt_cell(int nt_cell) {
	this->nt_cell = nt_cell;
}

void DenseTrajectories::set_scale_num(int scale_num) {
	this->scale_num = scale_num;
}

void DenseTrajectories::set_init_gap(int init_gap) {
	this->init_gap = init_gap;
}

void DenseTrajectories::set_export_header(bool use_header) {
	this->export_stats = use_header;
}

void DenseTrajectories::set_export_trajectories(bool export_trajectories) {
	this->export_tracklets = export_trajectories;
}

void DenseTrajectories::set_export_hog(bool export_hog) {
	this->export_hog = export_hog;
}

void DenseTrajectories::set_export_hof(bool export_hof) {
	this->export_hof = export_hof;
}

void DenseTrajectories::set_export_mbhx(bool export_mbhx) {
	this->export_mbhx = export_mbhx;
}

void DenseTrajectories::set_export_mbhy(bool export_mbhy) {
	this->export_mbhy = export_mbhy;
}

void DenseTrajectories::set_export_mbh(bool export_mbh) {
	this->export_mbh_whole = export_mbh;
}



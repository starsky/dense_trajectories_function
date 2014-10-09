#include "DenseTrackLib.h"
#include <time.h>

bool arg_parse(int argc, char** argv, DenseTrajectories& dt);

int main(int argc, char** argv)
{
	VideoCapture capture;
	DenseTrajectories processor;
	char* video = argv[1];
	int flag = arg_parse(argc, argv, processor);
	capture.open(video);
	if(!capture.isOpened()) {
		fprintf(stderr, "Could not initialize capturing..\n");
		return -1;
	}
	processor.initialize_dense_track();
	while(true) {
		// get a new frame
		std::vector<cv::Mat > featuresVect;
		Mat frame;
		capture >> frame;
		if(frame.empty())
			break;
		processor.process_frame(frame, &featuresVect);
		processor.printMat(featuresVect);
	}

	return 0;
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

bool arg_parse(int argc, char** argv, DenseTrajectories& dt) {
	int c;
	bool flag = false;
	char* executable = basename(argv[0]);
	while((c = getopt (argc, argv, "hS:E:L:W:N:s:t:A:I:")) != -1)
	switch(c) {
		case 'S':
		dt.set_start_frame(atoi(optarg));
		flag = true;
		break;
		case 'E':
		dt.set_end_frame(atoi(optarg));
		flag = true;
		break;
		case 'L':
		dt.set_track_length(atoi(optarg));
		break;
		case 'W':
		dt.set_min_distance(atoi(optarg));
		break;
		case 'N':
		dt.set_patch_size(atoi(optarg));
		break;
		case 's':
		dt.set_nxy_cell(atoi(optarg));
		break;
		case 't':
		dt.set_nt_cell(atoi(optarg));
		break;
		case 'A':
		dt.set_scale_num(atoi(optarg));
		break;
		case 'I':
		dt.set_init_gap(atoi(optarg));
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

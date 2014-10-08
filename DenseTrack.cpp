#include "DenseTrackLib.h"
#include <time.h>

int main(int argc, char** argv)
{
	VideoCapture capture;
	DenseTrajectories processor;
	char* video = argv[1];
	int flag = processor.arg_parse(argc, argv);
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


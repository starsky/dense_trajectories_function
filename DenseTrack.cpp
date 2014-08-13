#include "DenseTrackLib.h"
#include <time.h>

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

	return 0;
}


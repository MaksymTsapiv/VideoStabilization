#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/videostab/inpainting.hpp>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path/to/video>" << std::endl;
        return 1;
    }

    std::string video_path {argv[1]};

    // open video from file
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        return -1;
    }

    cv::Mat frame;
    // iterate through video frames
    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        // inpainting
        cv::videostab::MotionInpainter inpainter;
        cv::Mat mask;
        inpainter.inpaint(1, frame, mask);

        // show result
        cv::imshow("frame", frame);
        cv::imshow("mask", mask);
        if (cv::waitKey(30) >= 0) {
            break;
        }
    }

    return 0;
}

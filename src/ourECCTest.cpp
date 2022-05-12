#include <opencv2/opencv.hpp>
#include "ourECC.h"

int main() {
    std::string video_path{"../../short_video.mp4"};
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::Mat mat1, mat2;
    cap.read(mat1);
    cap.read(mat2);

    // Affine transformation
    // initialize p
    double data[6] = {1.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    cv::Mat params = cv::Mat(6, 1, CV_64FC1, &data);

//    ourFindTransformECC(mat1, mat2, params);


}
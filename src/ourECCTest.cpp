#include <opencv2/opencv.hpp>
#include "ourECC.h"

#define DEBUG

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

    cv::cvtColor(mat1, mat1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(mat2, mat2, cv::COLOR_BGR2GRAY);

#ifdef DEBUG
    std::cout << "Size of mat1: " << mat1.size() << std::endl;
    std::cout << "Size of mat2: " << mat2.size() << std::endl;
#endif

    // Affine transformation
    // initialize p
    double data[6] = {1.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    cv::Mat params = cv::Mat(6, 1, CV_64FC1, &data);

    ourFindTransformECC(mat1, mat2, params);


}
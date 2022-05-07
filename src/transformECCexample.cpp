#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>

double read_all_frames(const std::string &video_path, std::vector<cv::Mat> &out_frames);
void write_to_file(const std::string &path, const std::vector<cv::Mat> &frames, double fps);
std::vector<cv::Mat> get_warp_stack(std::vector<cv::Mat> frames);

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path/to/video>" << std::endl;
        return 1;
    }

    std::vector<cv::Mat> frames;
    double fps = read_all_frames(argv[1], frames);

    std::vector<cv::Mat> warp_stack = get_warp_stack(frames);

    cv::Mat curInverse = cv::Mat::eye(3, 3, CV_32F);
    for (size_t i = 1; i < frames.size(); ++i) {
        curInverse *= warp_stack[i-1].inv();
        cv::Mat transformed_frame;
        cv::warpAffine(frames[i], transformed_frame,
                       curInverse(cv::Range(0, 2), cv::Range::all()) / curInverse.at<float>(2, 2),
                       frames[i].size());
        frames[i] = transformed_frame;
    }

    write_to_file("stabilized.avi", frames, fps);

    return 0;
}

std::vector<cv::Mat> get_warp_stack(std::vector<cv::Mat> frames) {
    cv::Mat prevGrayMat;
    cv::cvtColor(frames[0], prevGrayMat, cv::COLOR_BGR2GRAY);

    std::vector<cv::Mat> warp_stack;

    for (size_t i = 1; i < frames.size(); ++i) {
        cv::Mat curGrayMat;
        cv::cvtColor(frames[i], curGrayMat, cv::COLOR_BGR2GRAY);
        cv::Mat transformMat = cv::Mat::eye(3, 3, CV_32FC1);

        cv::findTransformECC(prevGrayMat, curGrayMat, transformMat(cv::Range(0, 2), cv::Range::all()), cv::MOTION_AFFINE);

        warp_stack.push_back(std::move(transformMat));

        prevGrayMat = curGrayMat;
    }

    return warp_stack;
}

void write_to_file(const std::string &path, const std::vector<cv::Mat> &frames, double fps) {
    cv::VideoWriter video_writer(path, cv::VideoWriter::fourcc('M','J','P','G'), fps,
                                 frames[0].size());
    for (size_t idx = 0; idx < frames.size(); ++idx) {
        video_writer.write(frames[idx]);
    }
}

double read_all_frames(const std::string &video_path, std::vector<cv::Mat> &out_frames) {
    // open video from file
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::Mat curMat;
    while (cap.read(curMat)) {
        out_frames.push_back(std::move(curMat));
    }
    return cap.get(cv::CAP_PROP_FPS);
}

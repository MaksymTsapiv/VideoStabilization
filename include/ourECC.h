#ifndef CV_TEST_OURECC_H
#define CV_TEST_OURECC_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>

#include <vector>
#include <cstddef>

cv::Mat get_normalized_reference_intensities(const cv::Mat &templateImage,
                                                            std::vector<cv::KeyPoint> keypoints);

cv::Mat get_intensities(cv::Mat img, std::vector<cv::Point2d> points);

std::vector<cv::Point2d> get_warped_points(std::vector<cv::Point2d> points, cv::Mat params);

float interpolate(const cv::Mat &image, float x, float y);


cv::Mat get_column_zero_mean_jacobian(const cv::Mat &gradient_wrt_x, const cv::Mat &gradient_wrt_y,
                                      const cv::Mat &params, std::vector<cv::Point2d> &warped_points,
                                      const std::vector<cv::Point2d> &ref_points);

double ourFindTransformECC(const cv::InputArray &templateImage, const cv::InputArray &inputImage,
                           cv::Mat &params);

#endif //CV_TEST_OURECC_H

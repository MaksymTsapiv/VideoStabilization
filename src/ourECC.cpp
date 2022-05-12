#include "ourECC.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>

#include <vector>
#include <cstddef>

cv::Mat get_normalized_reference_intensities(const cv::Mat &templateImage,
                                             std::vector<cv::KeyPoint> keypoints) {
    cv::Mat intens{static_cast<int>(keypoints.size()), 1, CV_64F};
    for (size_t n_keypoint = 0; n_keypoint < keypoints.size(); ++n_keypoint) {
        intens.at<double>(n_keypoint, 0) = templateImage.at<double>(keypoints[n_keypoint].pt);
    }
    intens -= cv::mean(intens);
    intens /= cv::norm(intens);
    return intens;
}

cv::Mat get_intensities(cv::Mat img, std::vector<cv::Point2d> points) {
    cv::Mat intensities{static_cast<int>(points.size()), 1, CV_64F};
    for (size_t n_point = 0; n_point < points.size(); ++n_point) {
        intensities.at<double>(n_point, 0) = img.at<double>(points[n_point]);
    }
    return intensities;
}

std::vector<cv::Point2d> get_warped_points(std::vector<cv::Point2d> points, cv::Mat params) {
    std::vector<cv::Point2d> warped;
    for (int n_point = 0; points.size(); ++n_point) {
        cv::Point2d &cur_p = points[n_point];
        warped.emplace_back(
                cur_p.x * params.at<double>(0) + cur_p.y * params.at<double>(2) + params.at<double>(4),
                cur_p.x * params.at<double>(1) + cur_p.y * params.at<double>(3) + params.at<double>(5)
        );
    }
    return warped;
}

float interpolate(const cv::Mat &image, float x, float y)
{
    // Get the nearest integer pixel coords (xi;yi).
    int xi = cvFloor(x);
    int yi = cvFloor(y);

    float k1 = x-xi; // Coefficients for interpolation formula.
    float k2 = y-yi;

    int f1 = xi<image.cols - 1;  // Check that pixels to the right
    int f2 = yi<image.rows - 1; // and to down direction exist.

    double row10 = image.at<double>(yi, xi);
    double row11 = image.at<double>(yi, xi + 1);
    double row20 = image.at<double>(yi + 1, xi);
    double row21 = image.at<double>(yi + 1, xi + 1);

    // Interpolate pixel intensity.
    float interpolated_value = (1.0f-k1)*(1.0f-k2)*(float)row10 +
                               (f1 ? ( k1*(1.0f-k2)*(float)row11 ):0) +
                               (f2 ? ( (1.0f-k1)*k2*(float)row20 ):0) +
                               ((f1 && f2) ? ( k1*k2*(float)row21 ):0);

    return interpolated_value;
}


cv::Mat get_column_zero_mean_jacobian(const cv::Mat &gradient_wrt_x, const cv::Mat &gradient_wrt_y,
                                      const cv::Mat &params, std::vector<cv::Point2d> &warped_points,
                                      const std::vector<cv::Point2d> &ref_points) {
    cv::Mat jac{static_cast<int>(ref_points.size()), params.cols, CV_64F};
    for (int k = 0; k < jac.cols; ++k) {
        const cv::Point &cur_ref_point = ref_points[k];
        const cv::Point &cur_warped_point = warped_points[k];
        cv::Mat d_phi_d_p{2, 6};
        d_phi_d_p.at<double>(0, 0) = cur_ref_point.x;
        d_phi_d_p.at<double>(0, 1) = 0;
        d_phi_d_p.at<double>(0, 2) = cur_ref_point.y;
        d_phi_d_p.at<double>(0, 3) = 0;
        d_phi_d_p.at<double>(0, 4) = 1;
        d_phi_d_p.at<double>(0, 5) = 0;

        d_phi_d_p.at<double>(1, 0) = 0;
        d_phi_d_p.at<double>(1, 1) = cur_ref_point.x;
        d_phi_d_p.at<double>(1, 2) = 0;
        d_phi_d_p.at<double>(1, 3) = cur_ref_point.y;
        d_phi_d_p.at<double>(1, 4) = 0;
        d_phi_d_p.at<double>(1, 5) = 1;
        for (int n = 0; n < jac.rows; ++n) {
            jac.at<double>(k, n) = interpolate(gradient_wrt_x, cur_warped_point.x, cur_warped_point.y) * d_phi_d_p.at<double>(0, n) +
                                   interpolate(gradient_wrt_y, cur_warped_point.x, cur_warped_point.y) * d_phi_d_p.at<double>(1, n);
        }
    }
    for (int i = 0; i < jac.cols; ++i) {
        jac.col(i) -= cv::mean(jac.col(i));
    }
    return jac;
}

double ourFindTransformECC(const cv::InputArray &templateImage, const cv::InputArray &inputImage,
                           cv::Mat &params) {
    // Find points of interest
    int minHessian = 400;
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create(minHessian);
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(templateImage, keypoints);
    std::vector<cv::Point2d> target_area;
    for (const auto &keypoint : keypoints) {
        target_area.push_back(keypoint.pt);
    }

    // calculate normalized reference points
    cv::Mat r_normalized_int = get_normalized_reference_intensities(templateImage.getMat(), keypoints);



    // gradient of warped image
    cv::Mat gradient_wrt_x_warped, gradient_wrt_y_warped;
    cv::Scharr(inputImage.getMat(), gradient_wrt_x_warped, CV_64F, 1, 0, 5);
    cv::Scharr(inputImage.getMat(), gradient_wrt_y_warped, CV_64F, 0, 1, 5);

    while (true) {
        std::vector<cv::Point2d> warped_points = get_warped_points(target_area, params);
        cv::Mat zero_mean_warped_intens = get_intensities(inputImage.getMat(),
                                                          warped_points);
        cv::Mat jacobian = get_column_zero_mean_jacobian(gradient_wrt_x_warped, gradient_wrt_y_warped, params, warped_points, target_area);

        cv::Mat pg = jacobian * (jacobian.t() * jacobian).inv() * jacobian.t();

        cv::Mat l_mat = r_normalized_int.t() * zero_mean_warped_intens;
        cv::Mat r_mat = r_normalized_int.t() * pg * zero_mean_warped_intens;

        cv::Mat d_p;
        if (l_mat.at<double>(0, 0) > r_mat.at<double>(0, 0)) {
            cv::Mat denom = r_normalized_int.t() * zero_mean_warped_intens - r_normalized_int.t() * pg * zero_mean_warped_intens;
            cv::Mat diff = (zero_mean_warped_intens.t() * zero_mean_warped_intens - zero_mean_warped_intens.t() * pg * zero_mean_warped_intens) /
                           denom.at<double>(0, 0) - zero_mean_warped_intens;
            d_p = (jacobian.t() * jacobian).inv() * jacobian.t() * diff;
        } else {
            cv::Mat denom = r_normalized_int.t() * pg * r_normalized_int;
            cv::Mat lambda1_mat;
            cv::sqrt(zero_mean_warped_intens.t() * pg * zero_mean_warped_intens /
                     denom.at<double>(0, 0), lambda1_mat);
            double lambda1 = lambda1_mat.at<double>(0, 0);

            cv::Mat lambda2_mat = (r_normalized_int.t() * pg * zero_mean_warped_intens -
                                   r_normalized_int.t() * zero_mean_warped_intens) / denom;
            double lambda2 = lambda2_mat.at<double>(0, 0);

            double max_lambda = std::max(lambda1, lambda2);

            d_p = (jacobian.t() * jacobian).inv() * jacobian.t() *
                  (max_lambda * r_normalized_int - zero_mean_warped_intens);
        }

        params += d_p;
    }

    return -1;
}
#include "ourECC.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>

#include <vector>
#include <cstddef>

#define DEBUG

double meanOfDoubles(const cv::Mat &mat) {
    double summ = 0;
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            summ += mat.at<double>(i, j);
        }
    }
    return summ / mat.rows / mat.cols;
}

cv::Mat get_normalized_reference_intensities(const cv::Mat &templateImage,
                                             const std::vector<cv::Point2d> &target_area) {
    cv::Mat intens(static_cast<int>(target_area.size()), 1, CV_64F);
    for (size_t n_keypoint = 0; n_keypoint < target_area.size(); ++n_keypoint) {
        intens.at<double>(n_keypoint, 0) = interpolate(templateImage, target_area[n_keypoint].y, target_area[n_keypoint].x);
    }
    intens -= meanOfDoubles(intens);
    intens = intens / cv::norm(intens);
    return intens;
}

cv::Mat get_intensities(const cv::Mat &img, const std::vector<cv::Point2d> &points) {
    cv::Mat intensities(static_cast<int>(points.size()), 1, CV_64F);
    for (size_t n_point = 0; n_point < points.size(); ++n_point) {
        intensities.at<double>(n_point, 0) = interpolate(img, points[n_point].y, points[n_point].x);
    }
    return intensities;
}

std::vector<cv::Point2d> get_warped_points(std::vector<cv::Point2d> points, cv::Mat params) {
    std::vector<cv::Point2d> warped;
    for (int n_point = 0; n_point < points.size(); ++n_point) {
        cv::Point2d &cur_p = points[n_point];
        warped.emplace_back(
                cur_p.x * params.at<double>(0) + cur_p.y * params.at<double>(2) + params.at<double>(4),
                cur_p.x * params.at<double>(1) + cur_p.y * params.at<double>(3) + params.at<double>(5)
        );
    }
    return warped;
}

double interpolate(const cv::Mat &image, float y, float x)
{
    // Get the nearest integer pixel coords (xi;yi).
    int xi = cvFloor(x);
    int yi = cvFloor(y);

    float k1 = x-xi; // Coefficients for interpolation formula.
    float k2 = y-yi;

    int f1 = xi < image.cols - 1;  // Check that pixels to the right
    int f2 = yi < image.rows - 1; // and to down direction exist.

    double row10 = image.at<uint8_t>(yi, xi);
    double row11 = image.at<uint8_t>(yi, xi + 1);
    double row20 = image.at<uint8_t>(yi + 1, xi);
    double row21 = image.at<uint8_t>(yi + 1, xi + 1);

//    std::cout << row10 << ' ' << row11 << ' ' << row20 << ' ' << row21 << std::endl;

    // Interpolate pixel intensity.
    double interpolated_value = (1.0f - k1) * (1.0f - k2) * row10 +
                                (f1 ? (k1 * (1.0f - k2) * row11) : 0) +
                                (f2 ? ((1.0f-k1) * k2 * row20) : 0) +
                                ((f1 && f2) ? (k1 * k2 * row21 ) : 0);
    return interpolated_value;
}


cv::Mat get_column_zero_mean_jacobian(const cv::Mat &gradient_wrt_x, const cv::Mat &gradient_wrt_y,
                                      const cv::Mat &params, std::vector<cv::Point2d> &warped_points,
                                      const std::vector<cv::Point2d> &ref_points) {

    cv::Mat jac(static_cast<int>(ref_points.size()), params.rows * params.cols, CV_64F);
    for (int k = 0; k < jac.cols; ++k) {
        const cv::Point &cur_ref_point = ref_points[k];
        const cv::Point &cur_warped_point = warped_points[k];

//        std::vector<double>vec{(double)cur_ref_point.x, 0, (double)cur_ref_point.y, 0, 1, 0,
//                                 0, (double)cur_ref_point.x, 0, (double)cur_ref_point.y, 0, 1};
//        cv::Mat d_phi_d_p(2, 6, CV_64F, &vec[0]);
        cv::Mat d_phi_d_p(2, 6, CV_64F);
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
            jac.at<double>(k, n) = interpolate(gradient_wrt_x, cur_warped_point.y, cur_warped_point.x) * d_phi_d_p.at<double>(0, n) +
                                   interpolate(gradient_wrt_y, cur_warped_point.y, cur_warped_point.x) * d_phi_d_p.at<double>(1, n);
        }
    }
    for (int i = 0; i < jac.cols; ++i) {
        jac.col(i) -= meanOfDoubles(jac.col(i));
    }
    return jac;
}

double ourFindTransformECC(const cv::InputArray &templateImage, const cv::InputArray &inputImage,
                           cv::Mat &params) {
    // Find points of interest
    int minHessian = 100;
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create(minHessian);
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(templateImage, keypoints);
    std::vector<cv::Point2d> target_area;

    // Show points of interest
    cv::Mat img_keypoints;
    cv::drawKeypoints(templateImage, keypoints, img_keypoints);
    cv::imshow("Points of interest", img_keypoints);
    cv::imshow("Second image", inputImage);
    cv::waitKey(0);

    std::cout << "Coordinates of points: " << std::endl;
    for (const auto &keypoint : keypoints) {
        target_area.push_back(keypoint.pt);
        std::cout << target_area.back();
    }
    std::cout << std::endl;

#ifdef DEBUG
    std::cout << "Number of points of interest: " << target_area.size() << std::endl;
#endif

    // calculate normalized intensities of reference points
    cv::Mat r_normalized_int = get_normalized_reference_intensities(templateImage.getMat(), target_area);
    std::cout << "Normalized intensities of reference points: " << r_normalized_int << std::endl;

    // gradient of warped image
    cv::Mat gradient_wrt_x_warped, gradient_wrt_y_warped;
    cv::Scharr(inputImage.getMat(), gradient_wrt_x_warped, CV_64F, 1, 0, 5);
    cv::Scharr(inputImage.getMat(), gradient_wrt_y_warped, CV_64F, 0, 1, 5);
    std::cout << "Showing gradients" << std::endl;
    cv::imshow("Gradients with respect to x", gradient_wrt_x_warped);
    cv::imshow("Gradients with respect to y", gradient_wrt_y_warped);
    cv::waitKey(0);

    for (int iter = 0; iter < 100; ++iter) {
        std::cout << "Iteration #" << iter << std::endl;
        std::vector<cv::Point2d> warped_points = get_warped_points(target_area, params);
        std::cout << "Warped points: " << std::endl;
        for (int i = 0; i < warped_points.size(); ++i) {
            std::cout << warped_points[i] << std::endl;
        }

        cv::Mat zero_mean_warped_intens = get_intensities(inputImage.getMat(),
                                                          warped_points);
        zero_mean_warped_intens -= meanOfDoubles(zero_mean_warped_intens);
        std::cout << "Zero mean warped intensities: " << zero_mean_warped_intens << std::endl;

        cv::Mat jacobian = get_column_zero_mean_jacobian(gradient_wrt_x_warped, gradient_wrt_y_warped, params, warped_points, target_area);
        std::cout << "jacobian " << jacobian.rows << " x " << jacobian.cols << " :" << jacobian << std::endl;

        cv::Mat pg = jacobian * (jacobian.t() * jacobian).inv() * jacobian.t();
        std::cout << "(jacobian.t() * jacobian).inv(): " << (jacobian.t() * jacobian).inv() << std::endl;

        cv::Mat l_mat = r_normalized_int.t() * zero_mean_warped_intens;
        cv::Mat r_mat = r_normalized_int.t() * pg * zero_mean_warped_intens;

        cv::Mat d_p;
        std::cout << l_mat.at<double>(0, 0) << " > " << r_mat.at<double>(0, 0) << std::endl;
        if (l_mat.at<double>(0, 0) > r_mat.at<double>(0, 0)) {
            cv::Mat denom = r_normalized_int.t() * zero_mean_warped_intens - r_normalized_int.t() * pg * zero_mean_warped_intens;
            cv::Mat diff = (zero_mean_warped_intens.t() * zero_mean_warped_intens - zero_mean_warped_intens.t() * pg * zero_mean_warped_intens) /
                           denom.at<double>(0, 0) - zero_mean_warped_intens;
            d_p = (jacobian.t() * jacobian).inv() * jacobian.t() * diff;
        } else {
            cv::Mat denom = r_normalized_int.t() * pg * r_normalized_int;
            std::cout << "denom: " << denom << std::endl;

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
        std::cout << "Params: " << params << std::endl;
    }

    return -1;
}
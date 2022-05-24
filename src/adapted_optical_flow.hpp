/*
 * Adapted version of OpenCV Dense Optical Flow (using Lucas-Kanade Pyramidal method),
 * because original is not working :(
 *
 * Initial Sources:
 *  https://github.com/opencv/opencv_contrib/blob/4.x/modules/videostab/src/optical_flow.cpp
 *  https://github.com/opencv/opencv_contrib/blob/4.x/modules/videostab/include/opencv2/videostab/optical_flow.hpp
 */
#ifndef OPENCV_VIDEOSTAB_OPTICAL_FLOW_MY_HPP
#define OPENCV_VIDEOSTAB_OPTICAL_FLOW_MY_HPP

#include "opencv2/videostab/optical_flow.hpp"
#include "opencv2/cudaarithm.hpp"

using namespace cv;
using namespace cv::videostab;

class CV_EXPORTS Adapted_DensePyrLkOptFlowEstimatorGpu
        : public PyrLkOptFlowEstimatorBase, public IDenseOptFlowEstimator
{
public:
    Adapted_DensePyrLkOptFlowEstimatorGpu() {
        CV_Assert(cuda::getCudaEnabledDeviceCount() > 0);
        optFlowEstimator_ = cuda::DensePyrLKOpticalFlow::create();
    }

    void run(
            InputArray frame0, InputArray frame1, InputOutputArray flowX, InputOutputArray flowY,
            OutputArray errors) CV_OVERRIDE {

        frame0_.upload(frame0.getMat());
        frame1_.upload(frame1.getMat());

        optFlowEstimator_->setWinSize(winSize_);
        optFlowEstimator_->setMaxLevel(maxLevel_);

        if (errors.needed() && false) {
            CV_Error(Error::StsNotImplemented,
                    "DensePyrLkOptFlowEstimatorGpu doesn't support errors calculation");
        }
        else {
            cuda::GpuMat flow;
            optFlowEstimator_->calc(frame0_, frame1_, flow);

            cuda::GpuMat flows[2];
            cuda::split(flow, flows);

            flowX_ = flows[0];
            flowY_ = flows[1];

            // set errors
            errors.create(flowX_.size(), CV_32F);
            errors.setTo(Scalar::all(0));
        }

        flowX_.download(flowX.getMatRef());
        flowY_.download(flowY.getMatRef());
    }

private:
    Ptr<cuda::DensePyrLKOpticalFlow> optFlowEstimator_;
    cuda::GpuMat frame0_, frame1_, flowX_, flowY_, errors_;
};

#endif

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/videostab/inpainting.hpp>
#include <opencv2/videostab/stabilizer.hpp>
#include <opencv2/videostab/optical_flow.hpp>
#include <opencv2/videostab/motion_stabilizing.hpp>

namespace vs = cv::videostab;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path/to/video>" << std::endl;
        return 1;
    }

    std::string video_path {argv[1]};

    // VideoFileSource
    cv::Ptr<vs::VideoFileSource> source_ptr = cv::makePtr<vs::VideoFileSource>(video_path);

    vs::TwoPassStabilizer two_pass_stab;

    // Choose and configure motion stabilizer
    vs::GaussianMotionFilter motion_filter{3 /* ??? */};
    cv::Ptr<vs::GaussianMotionFilter> motion_stab = cv::makePtr<vs::GaussianMotionFilter>(motion_filter);
    two_pass_stab.setMotionStabilizer(motion_stab);

    // Do we also need to set motion estimator?

    two_pass_stab.setFrameSource(source_ptr);


    vs::InpaintingPipeline *inpainters = new vs::InpaintingPipeline();

    cv::Ptr<vs::ConsistentMosaicInpainter> mosaic_inp = cv::makePtr<vs::ConsistentMosaicInpainter>();
    mosaic_inp->setStdevThresh(3 /* ??? */);
    inpainters->pushBack(mosaic_inp);

    cv::Ptr<vs::MotionInpainter> motion_inp = cv::makePtr<vs::MotionInpainter>();
    motion_inp->setDistThreshold(3 /* ??? */);
    inpainters->pushBack(motion_inp);

    inpainters->setRadius(3 /* ??? */);
    two_pass_stab.setInpainter(inpainters);

    // Iterate through frames and do stabilization
    // TODO: implement

    return 0;
}

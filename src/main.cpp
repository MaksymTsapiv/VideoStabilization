/*
 * Some stabilization pipeline ideas taken from: 
 *   https://github.com/didpurwanto/video-stabilization-using-openCV
 */
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/videostab/inpainting.hpp>
#include <opencv2/videostab/stabilizer.hpp>
#include <opencv2/videostab/motion_stabilizing.hpp>

#include "adapted_optical_flow.hpp"

namespace vs = cv::videostab;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path/to/video>" << std::endl;
        return 1;
    }

    std::string output_path = "output.mp4";
    std::string video_path {argv[1]};

    // VideoFileSource
    cv::Ptr<vs::VideoFileSource> source_ptr = cv::makePtr<vs::VideoFileSource>(video_path);
    double output_fps = source_ptr->fps();

    vs::TwoPassStabilizer two_pass_stab;
    two_pass_stab.setFrameSource(source_ptr);


    two_pass_stab.setEstimateTrimRatio(true);
    two_pass_stab.setTrimRatio(0.1);
    two_pass_stab.setCorrectionForInclusion(true);
    two_pass_stab.setBorderMode(cv::BORDER_REPLICATE);


    // Choose and configure motion stabilizer
    vs::GaussianMotionFilter motion_filter{15, -1.0 /* ??? */};
    cv::Ptr<vs::GaussianMotionFilter> motion_stab = cv::makePtr<vs::GaussianMotionFilter>(motion_filter);
    two_pass_stab.setMotionStabilizer(motion_stab);


    // Wobble suppressor
    cv::Ptr<vs::MoreAccurateMotionWobbleSuppressorBase> wobble_sup = cv::makePtr<vs::MoreAccurateMotionWobbleSuppressor>();
    wobble_sup = cv::makePtr<vs::MoreAccurateMotionWobbleSuppressorGpu>();
    wobble_sup->setPeriod(30 /* ??? */);
    two_pass_stab.setWobbleSuppressor(wobble_sup);


    // Configure inpainters
    vs::InpaintingPipeline *inpainters = new vs::InpaintingPipeline();
    cv::Ptr<vs::InpainterBase> inpainters_base(inpainters);

    // Mosaicing
    cv::Ptr<vs::ConsistentMosaicInpainter> mosaic_inp = cv::makePtr<vs::ConsistentMosaicInpainter>();
    mosaic_inp->setStdevThresh(10.0 /* ??? */);
    inpainters->pushBack(mosaic_inp);


    // Motion Inapinter
    cv::Ptr<vs::MotionInpainter> inp = cv::makePtr<vs::MotionInpainter>();
    inp->setDistThreshold(5.0 /* ??? */);

    cv::Ptr<vs::IDenseOptFlowEstimator> flowEstimator;
    flowEstimator = cv::makePtr<Adapted_DensePyrLkOptFlowEstimatorGpu>();
    inp->setOptFlowEstimator(flowEstimator);

    inpainters->pushBack(inp);

    inpainters->setRadius(15 /* ??? */);
    two_pass_stab.setInpainter(inpainters_base);


    cv::VideoWriter writer;
    cv::Mat stabilizedFrame;
    int nframes = 0;
    char file_name[100];

    // Iterate through frameso, stabilize them and write to new video file
    while ( !(stabilizedFrame = two_pass_stab.nextFrame()).empty() ) {
        nframes++;

        // init writer (once) and save stabilized frame
        if (!writer.isOpened())
            writer.open(output_path, cv::VideoWriter::fourcc('X','V','I','D'),
                        output_fps, stabilizedFrame.size());
        writer << stabilizedFrame;
    }

    return 0;
}

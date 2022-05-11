#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/videostab/inpainting.hpp>
#include <opencv2/videostab/stabilizer.hpp>
#include <opencv2/videostab/motion_stabilizing.hpp>

namespace vs = cv::videostab;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path/to/video>" << std::endl;
        return 1;
    }

    std::string output_path = "output.mp4";
    double output_fps;

    std::string video_path {argv[1]};

    // VideoFileSource
    cv::Ptr<vs::VideoFileSource> source_ptr = cv::makePtr<vs::VideoFileSource>(video_path);

    output_fps = source_ptr->fps();

    vs::TwoPassStabilizer two_pass_stab;
    two_pass_stab.setFrameSource(source_ptr);

    // Choose and configure motion stabilizer
    vs::GaussianMotionFilter motion_filter{3 /* ??? */};
    cv::Ptr<vs::GaussianMotionFilter> motion_stab = cv::makePtr<vs::GaussianMotionFilter>(motion_filter);
    two_pass_stab.setMotionStabilizer(motion_stab);

    // Do we also need to set motion estimator?

    // Configure inpainters
    vs::InpaintingPipeline *inpainters = new vs::InpaintingPipeline();

    cv::Ptr<vs::ConsistentMosaicInpainter> mosaic_inp = cv::makePtr<vs::ConsistentMosaicInpainter>();
    mosaic_inp->setStdevThresh(3 /* ??? */);
    inpainters->pushBack(mosaic_inp);

    cv::Ptr<vs::MotionInpainter> motion_inp = cv::makePtr<vs::MotionInpainter>();
    motion_inp->setDistThreshold(3 /* ??? */);
    inpainters->pushBack(motion_inp);

    inpainters->setRadius(3 /* ??? */);
    two_pass_stab.setInpainter(inpainters);



    cv::VideoWriter writer;
    cv::Mat stabilizedFrame;
    int nframes = 0;
    char file_name[100];

    // for each stabilized frame
    while ( !(stabilizedFrame = two_pass_stab.nextFrame()).empty() ) {
        nframes++;

        // init writer (once) and save stabilized frame
        if (!writer.isOpened())
            writer.open(output_path, cv::VideoWriter::fourcc('X','V','I','D'), output_fps, stabilizedFrame.size());
        writer << stabilizedFrame;

        // show stabilized frame
        cv::imshow("stabilizedFrame", stabilizedFrame);
        sprintf(file_name, "%0.3d.tif", nframes);

        cv::imwrite(file_name, stabilizedFrame);

        char key = static_cast<char>(cv::waitKey(3));
        if (key == 27) {
            std::cout << std::endl;
            break;
        }
    }

    return 0;
}

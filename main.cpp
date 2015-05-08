#include "Panorama.h"

#include <vector>
#include <string>
#include <sstream>	// stringstream
#include <iomanip>	// setw setfill

void loadImages(std::vector<cv::Mat> & vecImgs);

int main()
{
	const float f = 595.0f, k1 = -0.15f, k2 = 0.001f;

	std::vector<cv::Mat> vecImgs;
	loadImages(vecImgs);

	cv::Mat panoImg;
	panorama(vecImgs, f, k1, k2, panoImg);
	
	cv::imshow("Final Panorama", panoImg);
	cv::waitKey(0);

	return 0;
}

void loadImages(std::vector<cv::Mat> & vecImgs)
{
	const unsigned nImgs = 18;
	//const std::string pre = "data/trees/trees_";
	const std::string pre = "data/campus/campus_";

	for (unsigned i = 0; i < nImgs; i++)
	{
		std::stringstream ss;
		ss << std::setw(3) << std::setfill('0') << i;
		std::string filePath = pre + ss.str() + ".jpg";

		vecImgs.push_back(cv::imread(filePath));
	}
}

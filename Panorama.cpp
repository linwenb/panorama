#include "Panorama.h"

#include <numeric>	// accumulate

//===================================================================
//==========================   PUBLIC   =============================
//===================================================================

void panorama(std::vector<cv::Mat> & vecImgs,
	const float & f, const float & k1, const float & k2,
	cv::Mat & panoImg)
{
	//--step.1 a series of input photos
	if (vecImgs.empty()) return;

	const unsigned nImgs = vecImgs.size();
	if (nImgs < 2)
	{
		panoImg = vecImgs.front().clone();
		return;
	}	// only one photo
		
	//--step.2 Warp images to spherical coordinate and Remove radial distortion
	for (unsigned i = 0; i < nImgs; i++)
	{
		warpSpherical(vecImgs[i], f, k1, k2);
	}

	//--step.3 Compute the features in images
	//--step.4 Match features between each pair of adjacent images
	//--step.5 Align each pair of adjacent images using RANSAC
	std::vector<cv::Mat> Hs(nImgs);
	Hs.front() = cv::Mat::eye(3, 3, CV_32FC1);	// identify matrix for first
	for (unsigned i = 1; i < nImgs; i++)
	{
		alignImage(vecImgs[i - 1], vecImgs[i], Hs[i]);
	}

	//--step.6 Blend the images into the final panorama
	blendPanorama(vecImgs, Hs, panoImg);

	//--step.7  Adjust image size if you want
	adjustSize(panoImg);
}

//===================================================================
//==========================   PRIVATE   ============================
//===================================================================

//------------------------  MAIN FUNCTIONS ---------------------------

void warpSpherical(cv::Mat & img, const float & f,
	const float & k1, const float & k2)
{
	// index after warping
	cv::Mat mapX(img.size(), CV_32FC1);
	cv::Mat mapY(img.size(), CV_32FC1);

	// camera at image center 
	float centX = 0.5f * img.cols;
	float centY = 0.5f * img.rows;

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			//-- Step 1: spherical coordinate, with known focal length
			float theta = atan2f(j - centX, f);
			float phi   = atan2f(i - centY, f);
			//float theta = (j - centX) / f;
			//float phi   = (i - centY) / f;

			//-- Step 2: corresponding Euclidean coordinate
			float xHat = sinf(theta) * cosf(phi);
			float yHat = sinf(phi);
			float zHat = cosf(theta) * cosf(phi);

			// skip Rotation!
			//-- Step 3: projection to the plane z = 1.
			xHat /= zHat;
			yHat /= zHat;

			//-- Step 4: correct radial Distortion, with known k1 and k2
			float r = xHat * xHat + yHat * yHat;

			float xt = xHat * (1.0f + k1 * r + k2 * r * r);
			float yt = yHat * (1.0f + k1 * r + k2 * r * r);

			//-- Step 5: convert back to planar image coordinate
			mapX.at<float>(i, j) = centX + xt * f;
			mapY.at<float>(i, j) = centY + yt * f;
		}
	}	// foreach pixel

	// update image
	cv::remap(img, img, mapX, mapY, CV_INTER_LINEAR,
		cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
}

void detectFeature(const cv::Mat & image,
	std::vector<cv::KeyPoint> & keypoints,
	cv::Mat & descriptor)
{
	cv::initModule_nonfree();
	
#if 1
	// SIFT features
	detectFeatureSIFT(image, keypoints, descriptor);
#else
	// or SURF features
	detectFeatureSURF(image, keypoints, descriptor);
#endif
}

void matchFeatures(const cv::Mat & img_1,
	const cv::Mat & img_2,
	std::vector<cv::KeyPoint> & keyPts_1,
	std::vector<cv::KeyPoint> & keyPts_2,
	std::vector<cv::DMatch>   & matches)
{
	//-- Step 1: Detect the keypoints
	//-- Step 2: Calculate descriptors (feature vectors)
	cv::Mat descriptors_1, descriptors_2;

	detectFeature(img_1, keyPts_1, descriptors_1);
	detectFeature(img_2, keyPts_2, descriptors_2);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	cv::FlannBasedMatcher   matcher;
	std::vector<cv::DMatch> origin_matches;
	matcher.match(descriptors_1, descriptors_2, origin_matches);

	//-- calculate minimum distance between keypoints
	float min_dist = FLT_MAX;
	const unsigned nOriMatches = origin_matches.size();
	for (unsigned i = 0; i < nOriMatches; i++)
	{
		float dist = origin_matches[i].distance;
		if (dist < min_dist) min_dist = dist;
	}

	//-- only "good" matches
	for (unsigned i = 0; i < nOriMatches; i++)
	{
		if (origin_matches[i].distance <= std::max(2 * min_dist, 0.02f))
		{
			matches.push_back(origin_matches[i]);
		}
	}
}

void alignImage(const cv::Mat & img_1, const cv::Mat & img_2, cv::Mat & H)
{
	//-- step.1 find matched pairs of feature points
	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
	std::vector<cv::DMatch>   matches;
	matchFeatures(img_1, img_2, keypoints_1, keypoints_2, matches);

	std::vector<cv::Point2f> pts_1, pts_2;
	const unsigned nMatches = matches.size();
	for (unsigned i = 0; i < nMatches; i++)
	{
		//-- Get the keypoints from the good matches
		pts_1.push_back(keypoints_1[matches[i].queryIdx].pt);
		pts_2.push_back(keypoints_2[matches[i].trainIdx].pt);
	}

	//-- step.2 compute the Homography Matrix, from 2 to 1
	computeHomography(H, pts_2, pts_1);
}

void blendPanorama(const std::vector<cv::Mat> & vecImgs,
	std::vector<cv::Mat> & Hs, cv::Mat & panoImg)
{
	cv::Mat tempImg;
	//-- step.1 compute final size and update homography matrix
	computeSize(vecImgs, Hs, tempImg);

	//-- step.2 blend all images
	const unsigned nImgs = vecImgs.size();
	for (unsigned i = 0; i < nImgs; i++)
	{
		blendImage(vecImgs[i], Hs[i], tempImg);
	}

	//-- step.3 normalize color using weight value at 4th channel
	normalizeBlend(tempImg, panoImg);

	//-- step.4 compute affine deformation to solve vertical drift
	affineDeform(panoImg, vecImgs.front(), Hs.front(), vecImgs.back(), Hs.back());
}

//------------------------  HELPING FUNCTIONS ---------------------------

void detectFeatureSIFT(const cv::Mat & image,
	std::vector<cv::KeyPoint> & keypoints,
	cv::Mat & descriptor)
{
	// Detect the keypoints
	cv::SiftFeatureDetector().detect(image, keypoints);
	// Calculate descriptors (feature vectors)
	cv::SiftDescriptorExtractor().compute(image, keypoints, descriptor);
}

void detectFeatureSURF(const cv::Mat & image,
	std::vector<cv::KeyPoint> & keypoints,
	cv::Mat & descriptor)
{
	cv::SurfFeatureDetector().detect(image, keypoints);
	cv::SurfDescriptorExtractor().compute(image, keypoints, descriptor);
}

void computeHomography(cv::Mat & H,
	const std::vector<cv::Point2f> & pts_2,
	const std::vector<cv::Point2f> & pts_1)
{
	// RANSAC algorithm	
	const float thresh = 4.0f;

	unsigned count = 0;	
	std::vector<unsigned> inliers;

	// minimum subset size is 1, so simply go through each
	unsigned n = pts_1.size();
	for (unsigned i = 0; i < n; i++)
	{
		// only consider translation here
		// compute motion based on current subset data
		float tu = pts_1[i].x - pts_2[i].x;
		float tv = pts_1[i].y - pts_2[i].y;

		std::vector<unsigned> tempInliers;
		for (unsigned j = 0; j < n; j++)
		{
			float tx = pts_2[j].x + tu;
			float ty = pts_2[j].y + tv;

			float d = dist(tx, ty, pts_1[j].x, pts_1[j].y);
			if (d <= thresh)
			{
				tempInliers.push_back(j);
			}	// it is inlier, that agrees with this motion
		}

		if (count < tempInliers.size())
		{
			count = tempInliers.size();
			inliers.swap(tempInliers);
		}	// more inliers this time
	}

	// averages all translations
	float u = 0.0f, v = 0.0f;
	for (const auto i : inliers)
	{
		u += pts_1[i].x - pts_2[i].x;
		v += pts_1[i].y - pts_2[i].y;
	}
	u /= inliers.size();
	v /= inliers.size();

	// get the homography matrix
	H = cv::Mat::eye(3, 3, CV_32FC1);
	H.at<float>(0, 2) = u;
	H.at<float>(1, 2) = v;
}

float dist(const float & x1, const float & y1,
	const float & x2, const float & y2)
{
	// euclidean distance
	float x = x1 - x2;
	float y = y1 - y2;
	return sqrtf(x * x + y * y);
}

void computeSize(const std::vector<cv::Mat> & vecImgs,
	std::vector<cv::Mat> & Hs, cv::Mat & panoImg)
{
	const unsigned nImgs = vecImgs.size();
	// update homography matrix based on the first image
	for (unsigned i = 1; i < nImgs; i++)
	{		
		Hs[i] = Hs[i - 1] * Hs[i];
	}

	float min_x = FLT_MAX, max_x = 0.0f;
	float min_y = FLT_MAX, max_y = 0.0f;
	// compute boundary of final size
	for (unsigned i = 0; i < nImgs; i++)
	{
		const int w = vecImgs[i].cols;
		const int h = vecImgs[i].rows;

		// four corners in image
		std::vector<cv::Point2f> corners;

		corners.push_back(cv::Point2f(0.0f, 0.0f));
		corners.push_back(cv::Point2f(0.0f, h - 1.0f));
		corners.push_back(cv::Point2f(w - 1.0f, 0.0f));
		corners.push_back(cv::Point2f(w - 1.0f, h - 1.0f));

		// become four corners in panoImg
		cv::perspectiveTransform(corners, corners, Hs[i]);

		for (const auto & corner : corners)
		{
			if (corner.x < min_x) min_x = corner.x;
			if (corner.x > max_x) max_x = corner.x;
			if (corner.y < min_y) min_y = corner.y;
			if (corner.y > max_y) max_y = corner.y;
		}
	}

	// update homography matrix for final panorama
	cv::Mat transform = cv::Mat::eye(3, 3, CV_32FC1);

	transform.at<float>(0, 2) = -min_x;
	transform.at<float>(1, 2) = -min_y;

	for (unsigned i = 0; i < nImgs; i++)
	{
		Hs[i] = transform * Hs[i];
	}

	// create height and width for a large image
	int height = (int)(ceil(max_y) - floor(min_y));
	int width  = (int)(ceil(max_x) - floor(min_x));

	// 4th channel for weight values
	panoImg = cv::Mat(height, width, CV_32FC4);
}

void blendImage(const cv::Mat & img, cv::Mat & H, cv::Mat & panoImg)
{
	const float blendWidth = 100.0;

	int min_x, min_y, max_x, max_y;
	// compute boundary of img in final panorama
	imageBoundry(img, H, min_x, min_y, max_x, max_y);
	
	cv::Mat invH = H.inv();

	// size of panorama
	const int h = panoImg.rows;
	const int w = panoImg.cols;

	// size of img
	const float r = img.rows - 1.0f;
	const float c = img.cols - 1.0f;

	for (int i = min_y; i <= max_y; i++)
	{
		for (int j = min_x; j <= max_x; j++)
		{
			if (i < 0 || i >= h || j < 0 || j >= w)
			{
				continue;
			}	// (i, j) out of panoImg

			// panoImg(i, j) map to img(x, y)
			std::vector<cv::Point2f> pts;
			pts.push_back(cv::Point2f((float)j, (float)i));
			cv::perspectiveTransform(pts, pts, invH);
			cv::Point2f & pt = pts.front();

			if (pt.x < 0.0 || pt.x >= c || pt.y < 0.0 || pt.y >= r)
			{
				continue;
			}	// pt out of img

			cv::Vec3b color;
			// get color at img(x, y) using inter linear method
			computeColor(img, pt.x, pt.y, color);

			// set the weight value
			float weight = std::min(j - min_x, max_x - j) / blendWidth;
			if (weight > 1.0f) weight = 1.0f;

			// copy color with weight from img to pano
			cv::Vec4f & val = panoImg.at<cv::Vec4f>(i, j);

			val[0] += weight * color[0];
			val[1] += weight * color[1];
			val[2] += weight * color[2];
			val[3] += weight;
		}
	}	// foreach pixel, mapped from img, in panoImg
}

void imageBoundry(const cv::Mat & img, const cv::Mat & H,
	int & min_x, int & min_y, int & max_x, int & max_y)
{
	float min_xf = FLT_MAX, max_xf = 0.0f;
	float min_yf = FLT_MAX, max_yf = 0.0f;

	const int h = img.rows;
	const int w = img.cols;

	std::vector<cv::Point2f> corners;
	// four corners in image
	corners.push_back(cv::Point2f(0.0, 0.0));
	corners.push_back(cv::Point2f(0.0, h - 1.0));
	corners.push_back(cv::Point2f(w - 1.0, 0.0));
	corners.push_back(cv::Point2f(w - 1.0, h - 1.0));

	// become four corners in panoImg
	cv::perspectiveTransform(corners, corners, H);

	for (const auto & corner : corners)
	{
		if (corner.x < min_xf) min_xf = corner.x;
		if (corner.x > max_xf) max_xf = corner.x;
		if (corner.y < min_yf) min_yf = corner.y;
		if (corner.y > max_yf) max_yf = corner.y;
	}

	// convert to int
	min_x = (int)floor(min_xf);
	min_y = (int)floor(min_yf);
	max_x = (int)ceil (max_xf);
	max_y = (int)ceil (max_yf);
}

void computeColor(const cv::Mat & img,
	const float & x, const float & y, cv::Vec3b & color)
{
	// x and y not int
	// compute color from 4 nearest pixels

	int xf = (int)floor(x);
	int yf = (int)floor(y);
	int xc = xf + 1;
	int yc = yf + 1;
	
	std::vector<float> wts;
	// calculate distances
	wts.push_back(dist(x, y, xf, yf));
	wts.push_back(dist(x, y, xc, yf));
	wts.push_back(dist(x, y, xf, yc));
	wts.push_back(dist(x, y, xc, yc));

	// shorter distance, higher weight
	float wt_tot = std::accumulate(wts.begin(), wts.end(), 0.0f);
	for (float & wt : wts) wt = wt_tot - wt;

	// normalize the values
	wt_tot = std::accumulate(wts.begin(), wts.end(), 0.0f);
	for (float & wt : wts) wt /= wt_tot;

	// copy color value with weights
	for (unsigned i = 0; i < 3; i++)
	{
		float colVal = 0.0f;

		colVal += wts[0] * img.at<cv::Vec3b>(yf, xf)[i];
		colVal += wts[1] * img.at<cv::Vec3b>(yf, xc)[i];
		colVal += wts[2] * img.at<cv::Vec3b>(yc, xf)[i];
		colVal += wts[3] * img.at<cv::Vec3b>(yc, xc)[i];

		color[i] = (uchar)colVal;
	}	// foreach RBG
}

void normalizeBlend(cv::Mat & tempImg, cv::Mat & panoImg)
{
	// tempImg has weighted color values from several images
	// normalize colors

	const int h = tempImg.rows;
	const int w = tempImg.cols;

	// create a same size
	panoImg = cv::Mat(h, w, CV_8UC3);

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			cv::Vec4f & val   = tempImg.at<cv::Vec4f>(i, j);
			cv::Vec3b & color = panoImg.at<cv::Vec3b>(i, j);

			if (val[3])
			{
				color[0] = (uchar)(val[0] / val[3]);
				color[1] = (uchar)(val[1] / val[3]);
				color[2] = (uchar)(val[2] / val[3]);
			}	// non-zero
			else
			{
				color[0] = (uchar)val[0];
				color[1] = (uchar)val[1];
				color[2] = (uchar)val[2];
			}			
		}
	}	// foreach pixel

	// clear tempImg
	tempImg.release();
}

void affineDeform(cv::Mat & panoImg,
	const cv::Mat & img_0, const cv::Mat & H_0,
	const cv::Mat & img_n, const cv::Mat & H_n)
{
	// use affine deformation to solve vertical drift

	// get the center point at the first image
	std::vector<cv::Point2f> pts;
	pts.push_back(cv::Point2f(0.5f * img_0.cols, 0.5f * img_0.rows));
	cv::perspectiveTransform(pts, pts, H_0);
	cv::Point2f pt_init = pts.front();
	
	// center point at the last image
	pts.clear();
	pts.push_back(cv::Point2f(0.5f * img_n.cols, 0.5f * img_n.rows));
	cv::perspectiveTransform(pts, pts, H_n);
	cv::Point2f pt_final = pts.front();
	
	// compute affine deformation matrix
	cv::Mat A = cv::Mat::eye(3, 3, CV_32FC1);
	A.at<float>(1, 0) = -(pt_final.y - pt_init.y) / (pt_final.x - pt_init.x);
	A.at<float>(1, 2) = -pt_init.x * A.at<float>(1, 0);

	cv::Mat tempImg;
	// update image with affine deformation
	cv::warpPerspective(panoImg, tempImg, A, panoImg.size());

	// remove the useless blank space
	int diff = (int)ceil(abs(pt_final.y - pt_init.y));
	cv::Mat subImg(tempImg, cv::Rect(0, diff, tempImg.cols, tempImg.rows - diff));
	subImg.copyTo(panoImg);
}

void adjustSize(cv::Mat & panoImg)
{
	// make final panorama smaller with fixed aspect ratio
	const int maxWidth = 2000;
	if (panoImg.cols > maxWidth)
	{
		int height = maxWidth * panoImg.rows / panoImg.cols;
		cv::resize(panoImg, panoImg, cv::Size(maxWidth, height));
	}
}

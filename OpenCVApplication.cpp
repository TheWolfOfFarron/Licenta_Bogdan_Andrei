// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <limits.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <stack>
#include <numeric>




wchar_t* projectPath;





Mat etichetare(const Mat src, float distance, int& nretichete);
int calculateCenterLabel(Mat src, Mat labels);



struct BiomarkerResults {
	double vesselDensity;
	double meanIntensity;
	double stdIntensity;
	std::vector<double> areas;
	std::vector<double> perimeters;
};


BiomarkerResults extractBiomarkers(const cv::Mat& image) {
	BiomarkerResults results;
	// Find contours
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(image, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	// Calculate areas and perimeters
	for (const auto& contour : contours) {
		results.areas.push_back(cv::contourArea(contour));
		results.perimeters.push_back(cv::arcLength(contour, true));
	}

	// Calculate vessel density
	double vesselArea = cv::countNonZero(image);
	double totalArea = image.total();
	results.vesselDensity = vesselArea / totalArea;

	// Calculate mean and standard deviation of intensity
	cv::Scalar mean, stddev;
	cv::meanStdDev(image, mean, stddev);
	results.meanIntensity = mean[0];
	results.stdIntensity = stddev[0];

	// Branching points detection (simple approach for demonstration)
	
	

	return results;



}



Mat vascularSeg(Mat img) {
	cv::threshold(img, img, 127, 255, cv::THRESH_TRIANGLE);
	cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat temp;
	cv::Mat eroded;

	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

	bool done;
	do
	{
		cv::erode(img, eroded, element);
		cv::dilate(eroded, temp, element); // temp = open(img)
		cv::subtract(img, temp, temp);
		cv::bitwise_or(skel, temp, skel);
		eroded.copyTo(img);

		done = (cv::countNonZero(img) == 0);
	} while (!done);

	Mat s1;
	 element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(2, 2));
	cv::erode(skel, s1, element);
	cv::erode(s1, s1, element);
	cv::erode(s1, s1, element);
	cv::dilate(s1, s1, element);
	cv::dilate(s1, s1, element);
	cv::dilate(s1, s1, element);
	imshow("test erode", s1);
	return skel;
}





void prepoc(Mat img) {
	Mat minm = img.clone();
	cv::Mat filteredImage1=img.clone();
	Mat ves = vascularSeg(img);

	imshow("vesela", ves);
	Mat min;
	cv::subtract(minm, ves, min);
	imshow("min242", min);


	cv::Mat filteredImage;
	cv::GaussianBlur(filteredImage1, filteredImage, cv::Size(5, 5), 0.6);
	imshow("filteredImage", filteredImage);



	imshow("min", min);
	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

	min = vascularSeg(min);
	cv::subtract(min, ves, min);

	imshow("min dilate", min);

	//waitKey();
}



cv::Mat removeSmallStructures(const cv::Mat& src, int size) {
	cv::Mat se = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size));
	cv::Mat opened;
	cv::morphologyEx(src, opened, cv::MORPH_OPEN, se); // Remove small objects
	return opened;
}



// Function to preprocess the image
cv::Mat preprocess(const cv::Mat& src) {
	cv::Mat gray;
	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY); // Convert to grayscale
	return gray;
}

// Function to enhance the vessels
cv::Mat enhanceVessels(const cv::Mat& src) {
	cv::Mat enhanced;
	// Here we use a combination of median filter and adaptive histogram equalization (CLAHE)
	cv::medianBlur(src, enhanced, 7); 
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	clahe->setClipLimit(4);
	clahe->apply(enhanced, enhanced); // Enhance contrast
	return enhanced;
}


cv::Mat extractVessels(const cv::Mat& src) {
	cv::Mat vessels;
	cv::adaptiveThreshold(src, vessels, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
		cv::THRESH_BINARY_INV, 11, 2);
	return vessels;
}

cv::Mat keepLargeStructures(const cv::Mat& src) {
	cv::Mat largeVessels;
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::morphologyEx(src, largeVessels, cv::MORPH_CLOSE, kernel);
	return largeVessels;
}


cv::Mat keepEdgeVessels(const cv::Mat& src) {
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(src.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat edgeVessels = cv::Mat::zeros(src.size(), CV_8UC1);

	for (size_t i = 0; i < contours.size(); i++) {
		std::vector<cv::Point> polygon;
		cv::approxPolyDP(contours[i], polygon, 1.0, true);

		bool startsFromEdge = false;
		for (const cv::Point& pt : polygon) {
			if (pt.x == 0 || pt.y == 0 || pt.x == src.cols - 1 || pt.y == src.rows - 1) {
				startsFromEdge = true;
				break;
			}
		}

		if (startsFromEdge && cv::contourArea(contours[i]) > 100) { // Threshold for "large" is set to 100 here
			cv::drawContours(edgeVessels, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);
		}
	}

	return edgeVessels;
}

cv::Mat keepEdgeVesselsv2(const cv::Mat& src) {
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(src.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat edgeVessels = cv::Mat::zeros(src.size(), CV_8UC1);

	for (size_t i = 0; i < contours.size(); i++) {
		bool startsFromEdge = false;
		for (const cv::Point& pt : contours[i]) {
			if (pt.x <= 1 || pt.y <= 1 || pt.x >= src.cols - 2 || pt.y >= src.rows - 2) {
				startsFromEdge = true;
				break;
			}
		}

		if (startsFromEdge && cv::contourArea(contours[i]) > 60) { // Threshold for "large" is set to 100 here
			cv::drawContours(edgeVessels, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);
		}
	}

	cv::Mat dilatedEdgeVessels;
	cv::dilate(edgeVessels, dilatedEdgeVessels, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10)));
	cv::dilate(dilatedEdgeVessels, dilatedEdgeVessels, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10)));

	// Use the dilated image as a mask to include original contours that are close to the edge vessels
	cv::Mat finalVessels;
	src.copyTo(finalVessels, dilatedEdgeVessels);

	return finalVessels;
}

Mat contrast(const Mat grayImage,double alpha) {



	double meanIntensity = cv::mean(grayImage)[0];

	//double alpha = 2.0;  // You can adjust this value to control contrast
	cv::Mat contrastImage = grayImage.clone();
	contrastImage = alpha * (contrastImage - meanIntensity) + meanIntensity;


	return  contrastImage;
}

Mat removeSmallComponentsNoise(const Mat src,int size) {
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(src.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat edgeVessels = cv::Mat::zeros(src.size(), CV_8UC1);


	for (size_t i = 0; i < contours.size(); i++) {
		if (cv::contourArea(contours[i]) > size) { // Threshold for "large" is set to 100 here
			cv::drawContours(edgeVessels, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);
		}
	}

	return edgeVessels;

}

std::vector<Mat> getRegions(const Mat src, const Mat graySrc) {
	std::vector<Mat> ret;
	std::vector<std::vector<Point>> contours;

	findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	// Draw bounding boxes for each contour
	for (size_t i = 0; i < contours.size(); ++i) {
		// Get bounding box
		cv::Rect boundingBox = cv::boundingRect(contours[i]);

		// Extract region
		ret.push_back(graySrc(boundingBox));
	}
	return ret;
}

Mat putBoxes(const Mat src) {
	std::vector<std::vector<Point>> contours;
	Mat image = src.clone();
	findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// Draw bounding boxes for each contour
	for (size_t i = 0; i < contours.size(); i++) {
		// Get the bounding rectangle of a contour
		Rect boundingRect = cv::boundingRect(contours[i]);
	
		// Draw the bounding rectangle on the original image
		rectangle(image, boundingRect, Scalar(255, 255, 255), 1);
	}

	return image;
}


//nu mere
void Classificare1(const Mat src, const Mat binarexit) {
	//statista/bayes... lab 5 srf  cu histrograma simpla

	imshow("src1wq", src);
	imshow("RIP!", binarexit);

	std::vector<float> u;
	std::vector<float> o;
	std::vector<float> fc;
	std::vector<float> fc1;
	std::vector<Mat> patches;
	std::vector<Mat> whitePatches;

	std::vector<Mat> histos;
	std::vector<Mat> histosW;

	int indexx = 0;
	int indexy = 0;

	float** covariatie = (float**)calloc(19 * 19, sizeof(float*));
	float** corelatie = (float**)calloc(19 * 19, sizeof(float*));

	for (int i = 0; i < 19 * 19; i++)
	{
		covariatie[i] = (float*)calloc(19 * 19, sizeof(float));
		corelatie[i] = (float*)calloc(19 * 19, sizeof(float));
	}

	for(int i=0;i<16;i++)
		for (int j = 0; j < 16; j++) {
			Mat patchesAux = Mat(19, 19, CV_8UC1);
			Mat whitePatchesAux = Mat(19, 19, CV_8UC1);
			int ok = 1;
			for (int l = 0; l < 19; l++)
			{
				for (int k = 0; k < 19; k++) {

					if (binarexit.at<uchar>(19 * i + l, 19 * j + k) == 255) {
						ok = 0;
					
					}

					patchesAux.at<uchar>(l, k) = src.at<uchar>(19 * i + l, 19 * j + k);
					whitePatchesAux.at<uchar>(l, k) = src.at<uchar>(19 * i + l, 19 * j + k);
				}
			

			}
			if (ok == 1) {
				patches.push_back(patchesAux);
				
			}
			else
			{
				whitePatches.push_back(whitePatchesAux);
			}
		}

	int channels[] = { 0 }; 
	int histSize[] = { 256 };
	float range[] = { 0, 256 };
	const float* ranges[] = { range };

	for (const auto& img: patches ) {
		Mat hist;
		 calcHist(&img, 1, channels, Mat(), hist, 1, histSize, ranges);
		 histos.push_back(hist);
	}


	for (const auto& img : whitePatches) {
		Mat hist;
		calcHist(&img, 1, channels, Mat(), hist, 1, histSize, ranges);
		histosW.push_back(hist);
	}

	Mat white_image(1, 1, CV_8UC1, Scalar(255));
	Mat white_hist;
	calcHist(&white_image, 1, channels, Mat(), white_hist, 1, histSize, ranges);
	
	for (int j = 0; j < histosW.size(); j++) {
		double min_distance = INT_MAX;
		int predicted_class = -1;

		for (int i = 0; i < histos.size(); ++i) {
			double dist = compareHist(histos[i], histosW[j], HISTCMP_BHATTACHARYYA);
			if (dist < min_distance) {
				min_distance = dist;
				predicted_class = 1;
			}
		}
		double dist = compareHist(white_hist, histosW[j], HISTCMP_BHATTACHARYYA);
		if (dist < min_distance) {
			predicted_class = 0;
		}

		std::cout << "whitepach " << j << "  " << predicted_class << '\n';
		if (predicted_class == 0) {
			imshow("whitepach " + j, whitePatches[j]);
		}
	}
}




Mat deteleLines(Mat src) {

	Mat image=src.clone();
	// Find contours in the binary image
	std::vector<std::vector<Point>> contours;
	findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// Iterate through the contours and remove the ones that resemble lines
	for (size_t i = 0; i < contours.size(); ++i) {
		// If the contour has very few vertices and a large aspect ratio, remove it
		if (contours[i].size() < 10) {
			Rect boundingRect = cv::boundingRect(contours[i]);
			float aspectRatio = static_cast<float>(boundingRect.width) / boundingRect.height;
			if (aspectRatio > 5) {  // Adjust this threshold based on your specific case
				// Fill the contour area with black color
				drawContours(image, contours, static_cast<int>(i), Scalar(0, 0, 0), FILLED);
			}
		}
	}
	return image;
	
}

std::vector<Mat> resizeVector(const std::vector<Mat> s) {
	std::vector<Mat> ret;
	for (int i = 0; i < s.size(); i++) {
		Mat aux;
		Size newSize(100, 100);
		resize(s[i], aux, newSize);

		// Convert pixel values to range [0, 1]
		aux.convertTo(aux, CV_32F, 1.0 / 255.0);
		ret.push_back(aux);
	}

	return ret;
}


void v2() {
	char fname[MAX_PATH];
	if (openFileDlg(fname)) {


		cv::Mat retina= imread(fname);
		Mat src = retina.clone();
		Mat src3 = retina.clone();
		Mat src4 = retina.clone();
		Mat retine = retina.clone();
		cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
		cv::cvtColor(retine, retine, cv::COLOR_BGR2GRAY);
		cv::cvtColor(src3, src3, cv::COLOR_BGR2GRAY);
		cv::cvtColor(src4, src4, cv::COLOR_BGR2GRAY);
		cv::Mat preprocessed = preprocess(retina);
		//imshow("preprocessed", preprocessed);

		cv::Mat enhanced = enhanceVessels(preprocessed);
		//imshow("enhanced", enhanced);

		
		cv::Mat vessels = extractVessels(enhanced);
		//imshow("vessels", vessels);

		cv::Mat largeVessels = keepLargeStructures(vessels);


		for(int i=0;i<src.rows;i++)
			for (int j = 0; j < src.cols; j++) {
				if (vessels.at<uchar>(i, j) == 0)
					src.at<uchar>(i, j) = 0;
			}

		std::cout << "ce?\n";

		//openFileDlg(fname);
		 
		Mat src2 = imread(fname,IMREAD_GRAYSCALE);
		for (int i = 0; i < src2.rows; i++)
			for (int j = 0; j < src2.cols; j++) {
				if (src2.at<uchar>(i, j) == 255)
					src3.at<uchar>(i, j) = 0;
			}
		Mat l1 = keepEdgeVessels(largeVessels);

		//imshow("largeVessels", largeVessels);
		for (int i = 0; i < largeVessels.rows; i++)
			for (int j = 0; j < largeVessels.cols; j++) {
				if (largeVessels.at<uchar>(i, j) == 255)
					largeVessels.at<uchar>(i, j) = 0;
				else
					largeVessels.at<uchar>(i, j) = 255;

			}
		imshow("invlargeVessels", largeVessels);

		//imshow("out", src);
		imshow("out2", src3);
		Mat l= keepEdgeVesselsv2(largeVessels);
		for (int i = 0; i < largeVessels.rows; i++)
			for (int j = 0; j < largeVessels.cols; j++) {
				if (l.at<uchar>(i, j) == 255)
					src4.at<uchar>(i, j) = 0;
			

			}
		//imshow("out3", src4);
		Mat doamneAjuta = src4.clone();
		GaussianBlur(src4, doamneAjuta, Size(5, 5), 0.8, 0.8);
		imshow("doamneAjuta", doamneAjuta);

		//imshow("inv", l);
		//imshow("notinv", l1);
		Mat contrasts = contrast(doamneAjuta,1.5); //inainte 2.0
		//imshow("contrast", contrasts);
		Mat thresholded= contrasts.clone();
		


		for (int i = 0; i < contrasts.rows; i++)
			for (int j = 0; j < contrasts.cols; j++) {
				if (contrasts.at<uchar>(i, j) >170)
					thresholded.at<uchar>(i, j) = 255;
				else
					thresholded.at<uchar>(i, j) = 0;

			}
		Mat aux;
		cv::dilate(thresholded, aux, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
		cv::erode(aux, thresholded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

		imshow("thresholded", thresholded);


		Mat important = thresholded.clone();

		important= removeSmallComponentsNoise(thresholded,100);

		imshow("RIP", important);
		

		std::cout << "rows: " << src.rows << "  cols " << src.cols<<'\n';

		//DEBUG 
		int ct = 0;
		int avreage=0;

		Mat textured = retine.clone();
		for(int i=0;i< retine.rows;i++)
			for (int j = 0; j < retine.cols; j++)
			{
				if (important.at<uchar>(i, j) == 255) {
					textured.at<uchar>(i, j) = retine.at<uchar>(i, j);
					ct++;
					avreage += retine.at<uchar>(i, j);
				}
				else
					textured.at<uchar>(i, j) = 0;
			}
		imshow("Textured", textured);

		 contrasts = contrast(textured,2.0);
		imshow("contrast", contrasts);

		contrasts = removeSmallComponentsNoise(contrasts, 100);

		Mat boxes = putBoxes(important);
		imshow("boxeex", boxes);


		for (int i = 0; i < contrasts.rows; i++)
			for (int j = 0; j < contrasts.cols; j++) {
				if (contrasts.at<uchar>(i, j) > avreage/ct)
					thresholded.at<uchar>(i, j) = 255;
				else
					thresholded.at<uchar>(i, j) = 0;

			}
		imshow("retrashold", thresholded);

	//	Classificare1(retine, important);
		//Mat s = deteleLines(thresholded);
		//imshow("retarded", s);

		int nretichete;
		Mat etic = etichetare(thresholded, 19, nretichete);
		cv::Mat etichtaImg = cv::Mat::zeros(retina.size(), CV_8UC3);
		int label = calculateCenterLabel(thresholded, etic);
		Mat Final = cv::Mat::zeros(retina.size(), CV_8UC1);
		std::cout << " label \n" << label << " label \n";
		for (int i = 0; i < Final.rows; i++)
			for (int j = 0; j < Final.cols; j++) {
				if (etic.at<double>(i, j) == label)
					Final.at<uchar>(i, j) = thresholded.at<uchar>(i, j);
			}

		imshow("FINALL", Final);


		std::vector<Mat> SmallPaches = getRegions(boxes, retine);
		std::vector<Mat> ReSmallPaches = resizeVector(SmallPaches);
		SmallPaches.clear();
		


		waitKey();

	}
}


/*
std::vector<std::vector<cv::Point>> filterContours(const cv::Mat& src) {
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(src.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	cv::Point center(src.cols / 2, src.rows / 2);
	double minDist = 1e6; // A large value to initialize the minimum distance
	int closestContourIdx = -1;

	// First, find the closest contour that starts from the edge
	for (size_t i = 0; i < contours.size(); i++) {
		for (const cv::Point& pt : contours[i]) {
			if (pt.x <= 10 || pt.y <= 10 || pt.x >= src.cols - 11 || pt.y >= src.rows - 11) {
				double dist = cv::norm(pt - center);
				if (dist < minDist) {
					minDist = dist;
					closestContourIdx = i;
				}
				break;
			}
		}
	}

	std::vector<std::vector<cv::Point>> filteredContours;
	for (size_t i = 0; i < contours.size(); i++) {
		if (static_cast<int>(i) != closestContourIdx) {
			filteredContours.push_back(contours[i]);
		}
	}

	return filteredContours;
}
*/


double distance(const Point& a, const Point& b) {
	return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

Mat processContours(const Mat& src, double areaThreshold) {
	std::vector<std::vector<Point>> contours;
	findContours(src.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	Mat edgeVessels = Mat::zeros(src.size(), CV_8UC1);

	Point imageCenter(src.cols / 2, src.rows / 2);
	int closestContourIndex = -1;
	double minDistanceToCenter = 1e+10;

	for (size_t i = 0; i < contours.size(); i++) {
		Moments mu = moments(contours[i]);
		if (mu.m00 == 0) continue; // Prevent division by zero
		Point2f centroid(mu.m10 / mu.m00, mu.m01 / mu.m00);
		bool startsFromEdge = false;
		for (const Point& pt : contours[i]) {
			if (pt.x <= 1 || pt.y <= 1 || pt.x >= src.cols - 2 || pt.y >= src.rows - 2) {
				startsFromEdge = true;
			}

		}
		double dist = distance(imageCenter, centroid);
		if (dist < minDistanceToCenter && startsFromEdge) {
			minDistanceToCenter = dist;
			closestContourIndex = static_cast<int>(i);
		}
	}

	for (size_t i = 0; i < contours.size(); i++) {
		bool startsFromEdge = false;
		for (const Point& pt : contours[i]) {
			if (pt.x <= 1 || pt.y <= 1 || pt.x >= src.cols - 2 || pt.y >= src.rows - 2) {
				startsFromEdge = true;
				break;
			}
		}

		if (startsFromEdge && static_cast<int>(i) == closestContourIndex && contourArea(contours[i]) > areaThreshold) {
			drawContours(edgeVessels, contours, static_cast<int>(i), Scalar(255), FILLED);
		}
		else {
			if (!startsFromEdge  && contourArea(contours[i]) > areaThreshold) {
				drawContours(edgeVessels, contours, static_cast<int>(i), Scalar(255), FILLED);
			}
		}
	}

	return edgeVessels;
}


cv::Point2f computeCentroid(const std::vector<cv::Point>& contour) {
	cv::Moments m = cv::moments(contour, false);
	return cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);
}

std::vector<std::vector<cv::Point>> mergeCloseContours(const std::vector<std::vector<cv::Point>>& contours, float threshold) {
	std::vector<std::vector<cv::Point>> mergedContours;
	std::vector<bool> merged(contours.size(), false);

	for (size_t i = 0; i < contours.size(); ++i) {
		if (merged[i]) continue;

		cv::Point2f centroid_i = computeCentroid(contours[i]);
		for (size_t j = i + 1; j < contours.size(); ++j) {
			if (merged[j]) continue;

			cv::Point2f centroid_j = computeCentroid(contours[j]);
			if (cv::norm(centroid_i - centroid_j) < threshold) {
				// Merge contours[i] and contours[j]
				std::vector<cv::Point> mergedContour;
				mergedContour.insert(mergedContour.end(), contours[i].begin(), contours[i].end());
				mergedContour.insert(mergedContour.end(), contours[j].begin(), contours[j].end());
				mergedContours.push_back(mergedContour);

				merged[i] = merged[j] = true;
				break; // only merging with one nearest contour
			}
		}

		if (!merged[i]) {
			mergedContours.push_back(contours[i]);
		}
	}

	return mergedContours;
}




std::vector<cv::Point2d> mergeCloseContours(const std::vector<std::vector<cv::Point>>& contours, double mergeDistance) {
	std::vector<cv::Point2d> centroids;
	for (const auto& contour : contours) {
		cv::Point2d centroid = computeCentroid(contour);
		if (centroid.x != -1 && centroid.y != -1) { 
			centroids.push_back(centroid);
		}
	}

	std::vector<bool> merged(centroids.size(), false);
	std::vector<cv::Point2d> mergedCentroids;

	for (size_t i = 0; i < centroids.size(); i++) {
		if (!merged[i]) {
			cv::Point2d mergedCentroid = centroids[i];
			int mergeCount = 1;
			for (size_t j = i + 1; j < centroids.size(); j++) {
				if (!merged[j] && cv::norm(centroids[i] - centroids[j]) < mergeDistance) {
					mergedCentroid.x += centroids[j].x;
					mergedCentroid.y += centroids[j].y;
					mergeCount++;
					merged[j] = true;
				}
			}
			mergedCentroid.x /= mergeCount;
			mergedCentroid.y /= mergeCount;
			mergedCentroids.push_back(mergedCentroid);
		}
	}

	std::cout << "Original centroids: " << centroids.size() << ", Merged centroids: " << mergedCentroids.size() << std::endl;
	
	return mergedCentroids;
}
// Main k-means clustering function
std::vector<int> clusterContours(const std::vector<std::vector<cv::Point>>& contours, int k, double mergeDistance) {
	std::vector<cv::Point2d> mergedCentroids = mergeCloseContours(contours, mergeDistance);
	if (mergedCentroids.empty()) {
		std::cerr << "No centroids to cluster" << std::endl;
		std::vector<int> s;
		return s;
	}
	// Convert to Mat for k-means
	cv::Mat data(mergedCentroids.size(), 2, CV_32F);
	for (size_t i = 0; i < mergedCentroids.size(); i++) {
		data.at<float>(i, 0) = static_cast<float>(mergedCentroids[i].x);
		data.at<float>(i, 1) = static_cast<float>(mergedCentroids[i].y);
	}

	cv::Mat labels, centers;

	if (static_cast<int>(mergedCentroids.size()) > 3)
		k = 3;
	else
		k = static_cast<int>(mergedCentroids.size());
	
	cv::kmeans(data, k, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);
	std::vector<int> clusterLabels(labels.rows);
	labels.row(0).copyTo(clusterLabels);

	return clusterLabels;
}


std::vector<std::vector<cv::Point>> mergeContours(const std::vector<std::vector<cv::Point>>& contours, double threshold) {
	std::vector<std::vector<cv::Point>> oldnewContours(contours.begin(), contours.end());

	bool ok = true;
	while (ok) {
		std::vector<cv::Point> centroids;

	for (const auto& contour : contours) {
		cv::Moments m = cv::moments(contour);
		centroids.push_back(cv::Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00)));
	}

	
	std::vector<std::vector<cv::Point>> newContours;

		ok = false;
		std::vector<bool> merged(oldnewContours.size(), false);
		for (size_t i = 0; i < oldnewContours.size(); ++i) {
			if (!merged[i]) {
				cv::Rect boundingBox = cv::boundingRect(oldnewContours[i]);

				for (size_t j = i + 1; j < oldnewContours.size(); ++j) {
					if (!merged[j] && distance(centroids[i], centroids[j]) < threshold) {
						boundingBox |= cv::boundingRect(oldnewContours[j]);
						merged[j] = true;
						ok = true;
					}
				}

				std::vector<cv::Point> newContour;
				newContour.push_back(boundingBox.tl());
				newContour.push_back(cv::Point(boundingBox.x + boundingBox.width, boundingBox.y));
				newContour.push_back(boundingBox.br());
				newContour.push_back(cv::Point(boundingBox.x, boundingBox.y + boundingBox.height));
				newContours.push_back(newContour);
				merged[i] = true;

			}
		}
		oldnewContours.assign(newContours.begin(), newContours.end());
		newContours.empty();
	}

	return oldnewContours;
}


void testScore(Mat testImg) {
	char fname[MAX_PATH];

	openFileDlg(fname);
	cv::Mat mask = imread(fname, IMREAD_GRAYSCALE);


	cv::Mat outputImage2 = cv::Mat::zeros(testImg.size(), CV_8UC3);



	float totalpoints = 0.0f;
	float truepoints = 0.0f;
	float falsepos = 0.0f;
	float falseneg = 0.0f;
	float IoU = 0.0f;
	float score = 0.0f;

	for (int i = 0; i < testImg.rows; i++) {
		for (int j = 0; j < testImg.cols; j++) {
			if (testImg.at<uchar>(i, j) == 255 && mask.at<uchar>(i, j) == 255) {
				outputImage2.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
				truepoints++;
			}
			if (mask.at<uchar>(i, j) == 255) {
				totalpoints++;
			}

			if (testImg.at<uchar>(i, j) == 255 && mask.at<uchar>(i, j) != 255) {
				outputImage2.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
				falsepos++;
			}

			if (testImg.at<uchar>(i, j) != 255 && mask.at<uchar>(i, j) == 255) {
				outputImage2.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
				falseneg++;
			}

		}
	}

	if (totalpoints != 0)
		score = truepoints * 100 / totalpoints;

	std::cout << "\nIOU: " << truepoints / (truepoints + falsepos + falseneg);

	imshow("score", outputImage2);
	std::cout << "\nscore: " << score << "\n";

	//Score E


}

int findClosestContourToCenter(const std::vector<std::vector<cv::Point>>& contours, const cv::Size& imageSize) {
	int closestIndex = -1;
	double minDistanceToCenter = DBL_MAX;

	cv::Point2f imageCenter(static_cast<float>(imageSize.width / 2.0), static_cast<float>(imageSize.height / 2.0));

	for (size_t i = 0; i < contours.size(); i++) {
		for (const auto& point : contours[i]) {
			double distanceToCenter = sqrt(pow(point.x-imageCenter.x,2)-pow(point.y - imageCenter.y, 2));
			if (distanceToCenter < minDistanceToCenter) {
				minDistanceToCenter = distanceToCenter;
				closestIndex = static_cast<int>(i);
			}
		}
	}

	return closestIndex;
}

bool isValidPixel(const Mat& binaryImage, int r, int c) {
	return r >= 0 && r < binaryImage.rows && c >= 0 && c < binaryImage.cols && binaryImage.at<uchar>(r, c) == 255;
}

Mat etichetare(const Mat src, float distance ,int &nretichete) {

	double distanceC = INT_MAX;
	cv::Point2f imageCenter(static_cast<float>(src.rows / 2.0), static_cast<float>(src.cols / 2.0));
	Mat etichete = cv::Mat::zeros(src.size(), CV_64F);
	int ct = 1;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 255 && etichete.at<double>(i,j)==0) {
				int min = INT_MAX;
				std::stack<std::pair<int, int>> pixelStack;
				pixelStack.push(std::make_pair(i, j));

				while (!pixelStack.empty()) {
					int curR = pixelStack.top().first;
					int curC = pixelStack.top().second;
					pixelStack.pop();

					// Atribuirea etichetei
					etichete.at<double>(curR, curC) = ct;
					int dis = 1;
					// Verificarea vecinilor pentru a vedea dacă sunt conectați
					if(curR > 0 && curR < src.rows-1 && curC > 0 && curC < src.cols-1)
					if (src.at<uchar>(curR - 1, curC) == 0 || src.at<uchar>(curR + 1, curC) == 0 || src.at<uchar>(curR, curC - 1) == 0 || src.at<uchar>(curR, curC + 1) == 0)
					{
						dis = distance;
					}
				
					for (int dr = -dis; dr <= dis; ++dr) {
						for (int dc = -dis; dc <= dis; ++dc) {
							int newR = curR + dr;
							int newC = curC + dc;
							if (isValidPixel(src, newR, newC) && etichete.at<double>(newR, newC) == 0) {
								pixelStack.push(std::make_pair(newR, newC));
							}
						}
					}
				}

				// Incrementarea etichetei pentru următorul obiect
				++ct;

				
			}
		}
	}
	nretichete = ct - 1;
	return etichete;
}

int calculateCenterLabel(Mat src, Mat labels) {
	cv::Point2f imageCenter(static_cast<float>(src.cols / 2.0), static_cast<float>(src.rows / 2.0));

	double min = INT_MAX;
	int label=0;

	for(int i=0;i<src.rows;i++)
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 255) {
				double distanceToCenter = sqrt(pow(j - imageCenter.x, 2) - pow(i - imageCenter.y, 2));
				if (min > distanceToCenter) {
					min = distanceToCenter;
					label = labels.at<double>(i, j);
				}
			}
		}

	return label;
}







void v3() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		cv::Mat retina = imread(fname,IMREAD_GRAYSCALE);
		cv::Mat colorRetina= imread(fname, IMREAD_COLOR);
		imshow("retina", retina);

		Mat binaryImage;

		cv::threshold(retina, binaryImage, 128, 255, THRESH_BINARY);
		imshow("enchanced", binaryImage);
		Mat out2 = removeSmallComponentsNoise(binaryImage,100);
		imshow("out2", out2);



		double areaThreshold = 60.0;
		Mat result = processContours(out2, areaThreshold);
		cv::imshow("Edge result", result);

		//std::vector<std::vector<cv::Point>> filteredContours = filterContours(out2);

		//// Draw the filtered contours
		//cv::Mat edgeVessels = cv::Mat::zeros(out2.size(), CV_8UC1);
		//for (size_t i = 0; i < filteredContours.size(); i++) {
		//	cv::drawContours(edgeVessels, filteredContours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);
		//}

		//cv::imshow("Edge Vessels", edgeVessels);

			// Extract contours from the binary image
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(result, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		// Set the number of clusters and the merge distance
		int k = 2; // For example, you want to cluster into 3 groups
		double mergeDistance = 50.0f;

		// Perform clustering on the contours
		std::vector<int> clusterLabels = clusterContours(contours, k, mergeDistance);
		std::vector<std::vector<cv::Point>> mergedContours = mergeContours(contours, mergeDistance);
		std::cout << clusterLabels.size() << "\n";


		int nretichete;
		Mat etic = etichetare(result, 19, nretichete);
		cv::Mat etichtaImg = cv::Mat::zeros(retina.size(), CV_8UC3);
		int label = calculateCenterLabel(result, etic);
		Mat Final= cv::Mat::zeros(retina.size(), CV_8UC1);
		std::cout << " label \n" << label << " label \n";
		for(int i=0;i<Final.rows;i++)
			for (int j = 0; j < Final.cols; j++) {
				if(etic.at<double>(i,j)==label)
				Final.at<uchar>(i, j) = result.at<uchar>(i, j);
			}

		imshow("FINALL", Final);
		std::default_random_engine gen;
		std::uniform_int_distribution<int>	d(0, 255);


		std::vector<cv::Vec3b> colors;
		std::random_device rd;
		std::mt19937 rng(rd());
		std::uniform_int_distribution<int> uni(0, 255);
		colors.push_back(cv::Vec3b(0,0,0));
		for (int i = 1; i <= nretichete; ++i) {
			colors.push_back(cv::Vec3b(uni(rng), uni(rng), uni(rng)));
		}

		for (int i = 0; i < etichtaImg.rows; i++) {
			for (int j = 0; j < etichtaImg.cols; j++) {
				if(result.at<uchar>(i,j) == 255)
				etichtaImg.at<Vec3b>(i, j)= colors[etic.at<double>(i,j)];
			}
		}
		std::cout << "12";

		imshow("etichete", etichtaImg);

		//DEBUG S

	
		// Create an output image
		cv::Mat outputImage = cv::Mat::zeros(retina.size(), CV_8UC3);
		std::cout << "trece";
		// Draw the contours wi th the cluster-specific random colors
		std::vector<Vec3b> colorss;

		for (size_t i = 0; i < mergedContours.size(); i++) {
			std::cout << "nr contours: " << mergedContours.size() << "\n";
			int randint = d(gen);
			Vec3b color = Vec3b(d(gen), d(gen), d(gen));
			colorss.push_back(color);
			cv::drawContours(outputImage, mergedContours, i, color,cv::FILLED);
			//cv::drawContours(outputImage, mergedContours, i, colors[clusterLabels[i]], cv::FILLED);
			String l = "Clusters" + i;
			cv::imshow(l, outputImage);


		}
		std::cout << "trece";
		cv::imshow("Clusters", outputImage);
		cv::Mat result3;

		colorRetina.copyTo(result3, outputImage);

		cv::imshow("Clusters2", result3);

		cv::imshow("result", result);

		//DEBUG E



		//Score S
		std::cout << "test1: \n";

		testScore(result); 
		std::cout << "\n";


		int closestIndex = findClosestContourToCenter(mergedContours, retina.size());

		// Create a mask image
		cv::Mat mask = cv::Mat::zeros(retina.size(), CV_8UC1);

		// Draw the closest contour on the mask
		cv::drawContours(mask, contours, closestIndex, cv::Scalar(255), cv::FILLED);

		// Extract only the pixels corresponding to the closest contour from the original image
		cv::Mat result2;
		retina.copyTo(result2, mask);

		// Display the result
		cv::imshow("Closest Contour", result2);

		std::cout << "last\n";
		testScore(Final);

		waitKey();
	}

}


int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		std::cout << "Menu: \n";
		std::cout << "13: v1 paper try \n";
		std::cout << "14: v2 algorithm  \n";
		std::cout << "15: test more img v2 \n";
		std::cout << "16: v3 anoimizate \n";


		scanf("%d",&op);
		switch (op)
		{
			
			case 13:
				char fname[MAX_PATH];
				if (openFileDlg(fname)) {
					Mat src = imread(fname, IMREAD_GRAYSCALE);
					imshow("src", src);
					prepoc(src);
				}
			case 14: 
				v2();
				break;
			case 15:
				for (int i = 0; i < 4; i++)
				{
					v2();
				}
				break;

			case 16:
				v3();
		}
	}
	while (op!=0);
	return 0;
}
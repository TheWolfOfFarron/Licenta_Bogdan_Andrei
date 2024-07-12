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
#include <fstream>
#include <iomanip>
#include <chrono>
#include <windows.h>



wchar_t* projectPath;





Mat etichetare(const Mat src, float distance, int& nretichete);
int calculateCenterLabel(Mat src, Mat labels);



struct BiomarkerResults {
	double vesselDensity;
	double meanIntensity;
	double stdIntensity;
	std::vector<double> areas;
	double totalAreas;
	std::vector<double> perimeters;
	double totalPerimeters;
};


BiomarkerResults extractBiomarkers(const cv::Mat& image) {
	BiomarkerResults results;
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(image, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	results.totalAreas = 0;
	results.totalPerimeters = 0;
	for (const auto& contour : contours) {
		results.areas.push_back(cv::contourArea(contour));
		results.totalAreas += cv::contourArea(contour);
		results.perimeters.push_back(cv::arcLength(contour, true));
		results.totalPerimeters = cv::arcLength(contour, true);
	}

	double vesselArea = cv::countNonZero(image);
	double totalArea = image.total();
	results.vesselDensity = vesselArea / totalArea;

	cv::Scalar mean, stddev;
	cv::meanStdDev(image, mean, stddev);
	results.meanIntensity = mean[0];
	results.stdIntensity = stddev[0];

	return results;



}

cv::Mat enhanceVessels(const cv::Mat& src) {
	cv::Mat enhanced;
	cv::medianBlur(src, enhanced, 7);
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	clahe->setClipLimit(4);
	clahe->apply(enhanced, enhanced); // contrast
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

		if (startsFromEdge && cv::contourArea(contours[i]) > 60) { 
			cv::drawContours(edgeVessels, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);
		}
	}

	cv::Mat dilatedEdgeVessels;
	cv::dilate(edgeVessels, dilatedEdgeVessels, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10)));
	cv::dilate(dilatedEdgeVessels, dilatedEdgeVessels, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10)));

	cv::Mat finalVessels;
	src.copyTo(finalVessels, dilatedEdgeVessels);

	return finalVessels;
}

Mat contrast(const Mat grayImage, double alpha) {
	double meanIntensity = cv::mean(grayImage)[0];

	//double alpha = 2.0;  
	cv::Mat contrastImage = grayImage.clone();
	contrastImage = alpha * (contrastImage - meanIntensity) + meanIntensity;


	return  contrastImage;
}

Mat removeSmallComponentsNoise(const Mat src, int size) {
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(src.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat edgeVessels = cv::Mat::zeros(src.size(), CV_8UC1);


	for (size_t i = 0; i < contours.size(); i++) {
		if (cv::contourArea(contours[i]) > size) { 
			cv::drawContours(edgeVessels, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);
		}
	}

	return edgeVessels;

}



Mat putBoxes(const Mat src) {
	std::vector<std::vector<Point>> contours;
	Mat image = src.clone();
	findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	for (size_t i = 0; i < contours.size(); i++) {
		Rect boundingRect = cv::boundingRect(contours[i]);
		rectangle(image, boundingRect, Scalar(255, 255, 255), 1);
	}

	return image;
}




void v2() {
	char fname[MAX_PATH];
	if (openFileDlg(fname)) {


		cv::Mat retina = imread(fname,IMREAD_GRAYSCALE);

		Mat src4 = retina.clone();
		Mat retine = retina.clone();



		cv::Mat enhanced = enhanceVessels(retina);


		cv::Mat vessels = extractVessels(enhanced);

		cv::Mat largeVessels = keepLargeStructures(vessels);

		
		imshow("invlargeVessels", largeVessels);

		Mat l = keepEdgeVesselsv2(largeVessels);
		for (int i = 0; i < largeVessels.rows; i++)
			for (int j = 0; j < largeVessels.cols; j++) {
				if (l.at<uchar>(i, j) == 255)
					src4.at<uchar>(i, j) = 0;


			}

		Mat Blur = src4.clone();
		GaussianBlur(src4, Blur, Size(5, 5), 0.8, 0.8);
		imshow("Gausian", Blur);


		Mat contrasts = contrast(Blur, 1.5); //inainte 2.0

		Mat thresholded = contrasts.clone();



		for (int i = 0; i < contrasts.rows; i++)
			for (int j = 0; j < contrasts.cols; j++) {
				if (contrasts.at<uchar>(i, j) > 170)
					thresholded.at<uchar>(i, j) = 255;
				else
					thresholded.at<uchar>(i, j) = 0;

			}
		Mat aux;
		cv::dilate(thresholded, aux, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
		cv::erode(aux, thresholded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

		imshow("thresholded", thresholded);


		Mat important = thresholded.clone();

		important = removeSmallComponentsNoise(thresholded, 100);

		imshow("After noise", important);




		int ct = 0;
		int avreage = 0;

		Mat textured = retine.clone();
		for (int i = 0; i < retine.rows; i++)
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

		contrasts = contrast(textured, 2.0);
		imshow("contrast", contrasts);

		contrasts = removeSmallComponentsNoise(contrasts, 100);

		Mat boxes = putBoxes(important);
		imshow("boxeex", boxes);


		for (int i = 0; i < contrasts.rows; i++)
			for (int j = 0; j < contrasts.cols; j++) {
				if (contrasts.at<uchar>(i, j) > avreage / ct)
					thresholded.at<uchar>(i, j) = 255;
				else
					thresholded.at<uchar>(i, j) = 0;

			}
		imshow("retrashold", thresholded);


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




		waitKey();

	}
}





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
		if (mu.m00 == 0) continue; 
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
			if (!startsFromEdge && contourArea(contours[i]) > areaThreshold) {
				drawContours(edgeVessels, contours, static_cast<int>(i), Scalar(255), FILLED);
			}
		}
	}

	return edgeVessels;
}




void testScoreDB(int index,Mat testImg, Mat mask) {

	


		float totalpoints = 0.0f;
		float truepoints = 0.0f;
		float falsepos = 0.0f;
		float falseneg = 0.0f;
		float IoU = 0.0f;
		float score = 0.0f;
		float trueNegatives = 0.0f;
		float Accuracy = 0.0f;
		float dice = 0.0f;
		float recall = 0.0f;
		float f1 = 0.0f;
		float precision = 0.0f;

		for (int i = 0; i < testImg.rows; i++) {
			for (int j = 0; j < testImg.cols; j++) {
				if (testImg.at<uchar>(i, j) == 255 && mask.at<uchar>(i, j) == 255) {
					truepoints++;
				}
				if (testImg.at<uchar>(i, j) == 0 && mask.at<uchar>(i, j) == 0) {
					trueNegatives++;
				}

				if (mask.at<uchar>(i, j) == 255) {
					totalpoints++;
				}

				if (testImg.at<uchar>(i, j) == 255 && mask.at<uchar>(i, j) != 255) {
					falsepos++;
				}

				if (testImg.at<uchar>(i, j) != 255 && mask.at<uchar>(i, j) == 255) {
					falseneg++;
				}
			}
		}
		std::cout << "Calculate variables\n";
		if (totalpoints != 0)
			score = truepoints * 100 / totalpoints;
		else
			if (truepoints == 0)
				score = 1;

		if ((truepoints + falsepos + falseneg) != 0)
			IoU = truepoints / (truepoints + falsepos + falseneg);
		else
			IoU = 1;
		if ((truepoints + falsepos + falseneg + trueNegatives) != 0)
			Accuracy = (truepoints + trueNegatives) / (truepoints + falsepos + falseneg + trueNegatives);
		else
			Accuracy = 1;

		if ((2 * truepoints + falsepos + falseneg) != 0)
			dice = (2 * truepoints) / (2 * truepoints + falsepos + falseneg);
		else
			dice = 1;
		if ((truepoints + falsepos) != 0)
			precision = (truepoints) / (truepoints + falsepos);
		else
			precision = 1;
		if ((truepoints + falseneg) != 0)
			recall = (truepoints) / (truepoints + falseneg);
		else
			recall = 1;
		if ((precision + recall)!=0)
			f1 = (2 * precision * recall) / (precision + recall);
		else
			f1 = 1;

		
		std::ofstream file;
		file.open("test_scores.csv", std::ios::out | std::ios::app);

		if (file.tellp() == 0) {
		//	file << "Index,Total Points,True Points,False Positives,False Negatives,True Negatives,IoU,Accuracy,Score\n";
		}

		file << std::fixed << std::setprecision(4)
			<< index+1 <<","
			<< totalpoints << ","
			<< truepoints << ","
			<< falsepos << ","
			<< falseneg << ","
			<< trueNegatives << ","
			<< IoU << ","
			<< Accuracy << ","
			<< dice << ","
			<< precision << ","
			<< recall << ","
			<< f1 << ","
			<< score << "\n";

		file.close();
	
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

void testScore1(Mat testImg) {

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
	float trueNegatives = 0.0f;
	float Accuracy = 0.0f;
	float dice = 0.0f;
	float recall = 0.0f;
	float f1 = 0.0f;
	float precision = 0.0f;

	for (int i = 0; i < testImg.rows; i++) {
		for (int j = 0; j < testImg.cols; j++) {
			if (testImg.at<uchar>(i, j) == 255 && mask.at<uchar>(i, j) == 255) {
				truepoints++;
			}
			if (testImg.at<uchar>(i, j) == 0 && mask.at<uchar>(i, j) == 0) {
				trueNegatives++;
			}

			if (mask.at<uchar>(i, j) == 255) {
				totalpoints++;
			}

			if (testImg.at<uchar>(i, j) == 255 && mask.at<uchar>(i, j) != 255) {
				falsepos++;
			}

			if (testImg.at<uchar>(i, j) != 255 && mask.at<uchar>(i, j) == 255) {
				falseneg++;
			}
		}
	}

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

	std::cout << "Calculate variables\n";
	if (totalpoints != 0)
		score = truepoints * 100 / totalpoints;
	else
		if (truepoints == 0)
			score = 1;

	if ((truepoints + falsepos + falseneg) != 0)
		IoU = truepoints / (truepoints + falsepos + falseneg);
	else
		IoU = 1;
	if ((truepoints + falsepos + falseneg + trueNegatives) != 0)
		Accuracy = (truepoints + trueNegatives) / (truepoints + falsepos + falseneg + trueNegatives);
	else
		Accuracy = 1;

	if ((2 * truepoints + falsepos + falseneg) != 0)
		dice = (2 * truepoints) / (2 * truepoints + falsepos + falseneg);
	else
		dice = 1;
	if ((truepoints + falsepos) != 0)
		precision = (truepoints) / (truepoints + falsepos);
	else
		precision = 1;
	if ((truepoints + falseneg) != 0)
		recall = (truepoints) / (truepoints + falseneg);
	else
		recall = 1;
	if ((precision + recall) != 0)
		f1 = (2 * precision * recall) / (precision + recall);
	else
		f1 = 1;
	// Write the values
	std::cout 
		<< IoU << ","
		<< Accuracy << ","
		<< dice << ","
		<< precision << ","
		<< recall << ","
		<< f1 << ","
		<< score << "\n";
}


int findClosestContourToCenter(const std::vector<std::vector<cv::Point>>& contours, const cv::Size& imageSize) {
	int closestIndex = -1;
	double minDistanceToCenter = DBL_MAX;

	cv::Point2f imageCenter(static_cast<float>(imageSize.width / 2.0), static_cast<float>(imageSize.height / 2.0));

	for (size_t i = 0; i < contours.size(); i++) {
		for (const auto& point : contours[i]) {
			double distanceToCenter = sqrt(pow(point.x - imageCenter.x, 2) - pow(point.y - imageCenter.y, 2));
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

Mat etichetare(const Mat src, float distance, int& nretichete) {

	double distanceC = INT_MAX;
	cv::Point2f imageCenter(static_cast<float>(src.rows / 2.0), static_cast<float>(src.cols / 2.0));
	Mat etichete = cv::Mat::zeros(src.size(), CV_64F);
	int ct = 1;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 255 && etichete.at<double>(i, j) == 0) {
				int min = INT_MAX;
				std::stack<std::pair<int, int>> pixelStack;
				pixelStack.push(std::make_pair(i, j));

				while (!pixelStack.empty()) {
					int curR = pixelStack.top().first;
					int curC = pixelStack.top().second;
					pixelStack.pop();

					etichete.at<double>(curR, curC) = ct;
					int dis = 1;
					if (curR > 0 && curR < src.rows - 1 && curC > 0 && curC < src.cols - 1)
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
	int label = 0;

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 255) {
				double distanceToCenter = sqrt(pow(j - imageCenter.x, 2) + pow(i - imageCenter.y, 2));
				if (min > distanceToCenter) {
					min = distanceToCenter;
					label = labels.at<double>(i, j);
				}
			}
		}

	return label;
}








void v4() {
	char fname[MAX_PATH];

	if (openFileDlg(fname)) {
		Mat retina = imread(fname, IMREAD_GRAYSCALE);
		imshow("retina", retina);


		Mat contrasted = contrast(retina, 2);
		imshow("contrasted", contrasted);


		Mat binaryImage;
		cv::threshold(contrasted, binaryImage, 140, 255, THRESH_BINARY);
		imshow("binary", binaryImage);


		Mat out2 = removeSmallComponentsNoise(binaryImage, 100);
		std::cout << "removeSmallComponentsNoise\n";
		imshow("smalComps", out2);



		double areaThreshold = 60.0;
		Mat result = processContours(out2, areaThreshold);
		std::cout << "processContours\n";
		imshow("processContours", result);


		int nretichete;
		Mat etic = etichetare(result, 19, nretichete);
		std::cout << "etichetare\n";


		std::default_random_engine gen;
		std::uniform_int_distribution<int>	d(0, 255);
		cv::Mat etichtaImg = cv::Mat::zeros(retina.size(), CV_8UC3);

		std::vector<cv::Vec3b> colors;
		std::random_device rd;
		std::mt19937 rng(rd());
		std::uniform_int_distribution<int> uni(0, 255);
		colors.push_back(cv::Vec3b(0, 0, 0));
		for (int i = 1; i <= nretichete; ++i) {
			colors.push_back(cv::Vec3b(uni(rng), uni(rng), uni(rng)));
		}

		for (int i = 0; i < etichtaImg.rows; i++) {
			for (int j = 0; j < etichtaImg.cols; j++) {
				if (result.at<uchar>(i, j) == 255)
					etichtaImg.at<Vec3b>(i, j) = colors[etic.at<double>(i, j)];
			}
		}
		std::cout << "12";

		imshow("etichete", etichtaImg);


		int label = calculateCenterLabel(result, etic);
		std::cout << "calculateCenterLabel\n";



		Mat Final = cv::Mat::zeros(retina.size(), CV_8UC1);
		std::cout << " label \n" << label << " label \n";
		for (int i = 0; i < Final.rows; i++)
			for (int j = 0; j < Final.cols; j++) {
				if (etic.at<double>(i, j) == label)
					Final.at<uchar>(i, j) = result.at<uchar>(i, j);

				imshow("Final", Final);



				std::cout << "Final\n";//12

				testScore1(Final);

				waitKey();

			}
	}

}

void saveBiomarkers(BiomarkerResults biomarkers){
	std::ofstream file;
	file.open("Images/test_scoresss.csv", std::ios::out );
	if (file.tellp() == 0) {
		//	file << "Index,Total Points,True Points,False Positives,False Negatives,True Negatives,IoU,Accuracy,Score\n";
	}

	// Write the values
	file << std::fixed << std::setprecision(4)
		<< "areas" << ","
		<< biomarkers.totalAreas << "\n"
		<< "meanIntensity" << ","
		<< biomarkers.meanIntensity << "\n"
		<< "perimeters" << ","
		<< biomarkers.totalPerimeters << "\n"
		<< "stdIntensity" << ","
		<< biomarkers.stdIntensity << "\n"
		<< "vesselDensity" << ","
		<< biomarkers.vesselDensity << "\n";


	file.close();

}


void appReady(Mat retina){

		
	imwrite("Images/retina.png", retina);

	Mat contrasted = contrast(retina, 2);
		
	imwrite("contrasted.png", contrasted);

	Mat binaryImage;
	cv::threshold(contrasted, binaryImage, 140, 255, THRESH_BINARY);
	
	imwrite("Images/binaryImage.png", binaryImage);

	Mat out2 = removeSmallComponentsNoise(binaryImage, 100);
	std::cout << "removeSmallComponentsNoise\n";
		
	imwrite("Images/smalComps.png", out2);


	double areaThreshold = 60.0;
	Mat result = processContours(out2, areaThreshold);
	std::cout << "processContours\n";
		
	imwrite("Images/processContours.png", result);

	int nretichete;
	Mat etic = etichetare(result, 19, nretichete);
	std::cout << "etichetare\n";


	std::default_random_engine gen;
	std::uniform_int_distribution<int>	d(0, 255);
	cv::Mat etichtaImg = cv::Mat::zeros(retina.size(), CV_8UC3);

	std::vector<cv::Vec3b> colors;
	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_int_distribution<int> uni(0, 255);
	colors.push_back(cv::Vec3b(0, 0, 0));
	for (int i = 1; i <= nretichete; ++i) {
		colors.push_back(cv::Vec3b(uni(rng), uni(rng), uni(rng)));
	}

	for (int i = 0; i < etichtaImg.rows; i++) {
		for (int j = 0; j < etichtaImg.cols; j++) {
			if (result.at<uchar>(i, j) == 255)
				etichtaImg.at<Vec3b>(i, j) = colors[etic.at<double>(i, j)];
		}
	}

	

	imwrite("Images/etichtaImg.png", etichtaImg);

	int label = calculateCenterLabel(result, etic);




	Mat Final = cv::Mat::zeros(retina.size(), CV_8UC1);
		
	for (int i = 0; i < Final.rows; i++)
		for (int j = 0; j < Final.cols; j++) {
			if (etic.at<double>(i, j) == label)
				Final.at<uchar>(i, j) = result.at<uchar>(i, j);
		}
	imwrite("Images/Final.png", Final);
	Mat biomarkers=cv::Mat::zeros(retina.size(), CV_8UC1);
	Mat Overlay=cv::Mat::zeros(retina.size(), CV_8UC3);
	for (int i = 0; i < Final.rows; i++)
		for (int j = 0; j < Final.cols; j++) {
			if (Final.at<uchar>(i, j) == 255) {
				biomarkers.at<uchar>(i, j) = retina.at<uchar>(i, j); \
				Overlay.at<Vec3b>(i, j) = Vec3b(retina.at<uchar>(i, j), 0, 0);
			}
			else {
				Overlay.at<Vec3b>(i, j) = Vec3b(retina.at<uchar>(i, j), retina.at<uchar>(i, j), retina.at<uchar>(i, j));
			}

		}
	imwrite("Images/trueImage.png", biomarkers);
	imwrite("Images/Overlay.png", Overlay);

	saveBiomarkers(extractBiomarkers(biomarkers));
}


void testDataset(){

	std::vector<Mat> retinaScan;
	std::vector<Mat> mask;
	
	char fname[MAX_PATH];

	auto start = std::chrono::high_resolution_clock::now();

	char folderName[MAX_PATH];
	char folderName2[MAX_PATH];


	if (openFolderDlg(folderName) == 0)
	{

	}
	if (openFolderDlg(folderName2) == 0)
	{

	}

	waitKey(3000000000);
	std::cout << folderName;
	std::cout << folderName2;

	int c = 0, index = 1, pos = 1;
	while (1) {
		sprintf(fname, "%s\\%d.png", folderName, index++);
		std::cout << fname << "\n";

		Mat img = imread(fname,IMREAD_GRAYSCALE);
		if (img.cols == 0) break;

		retinaScan.push_back(img);
		pos++;
	}
	index = 1;
	while (1) {
		sprintf(fname, "%s\\%d.png", folderName2, index++);
		Mat img = imread(fname, IMREAD_GRAYSCALE);
		if (img.cols == 0) break;

		mask.push_back(img);
		pos++;
	}

	std::cout << retinaScan.size()<<" " << mask.size();


	for (index = 0; index < retinaScan.size(); index++) {
		std::cout << "INDEX " << index << "\n";
		Mat retina = retinaScan[index];

		Mat contrasted = contrast(retina, 2);

		Mat binaryImage;
		cv::threshold(contrasted, binaryImage, 140, 255, THRESH_BINARY);
		Mat out2 = removeSmallComponentsNoise(binaryImage, 100);
		std::cout << "removeSmallComponentsNoise\n" ;
		double areaThreshold = 60.0;
		Mat result = processContours(out2, areaThreshold);
		std::cout << "processContours\n";

		int nretichete;
		Mat etic = etichetare(result, 19, nretichete);
		std::cout << "etichetare\n";

		cv::Mat etichtaImg = cv::Mat::zeros(retina.size(), CV_8UC3);
		int label = calculateCenterLabel(result, etic);
		std::cout << "calculateCenterLabel\n";

		Mat Final = cv::Mat::zeros(retina.size(), CV_8UC1);
		std::cout << " label \n" << label << " label \n";
		for (int i = 0; i < Final.rows; i++)
			for (int j = 0; j < Final.cols; j++) {
				if (etic.at<double>(i, j) == label)
					Final.at<uchar>(i, j) = result.at<uchar>(i, j);
			}
		std::cout << "Final\n";


		testScoreDB(index+118,Final,mask[index]);

	}
	auto end = std::chrono::high_resolution_clock::now();
	std::ofstream file;
	std::chrono::duration<double> duration = end - start;
	file.open("test_scores.csv", std::ios::out | std::ios::app);
	file << duration.count();

}



int main(int argc, char** argv)
{

	HWND hWnd = GetConsoleWindow();
	ShowWindow(hWnd, SW_HIDE);

	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	projectPath = _wgetcwd(0, 0);

	


	if (argc == 2) {
		cv::Mat image = cv::imread(argv[1], IMREAD_GRAYSCALE);
		appReady(image);
	}
	else
	{
		int op;
		do
		{
			system("cls");
			destroyAllWindows();
			std::cout << "Menu: \n";
			std::cout << "1: Algoritm pentru OCTA500\n";
			std::cout << "2: Testare algoritm OFTA-CLUJ  \n";



			scanf("%d", &op);
			switch (op)
			{
			case 1:
				v2();
				break;
			case 2:
				testDataset();
				break;
			case 3:
				v4();
				break;
			}
			
		} while (op != 0);
	}

	return 0;
}
// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

wchar_t* projectPath;










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



Mat minus(Mat img,Mat img2) {

	imshow("wtf", img);

	/*for(int i=0;i<img.rows;i++)
		for (int j = 0; j < img.cols; j++)
		{

			if (img2.at<uchar>(i, j)==255) {
				if (i >= 1 && i < img.rows - 1 && j >= 1 && j < img.cols - 1) {
					img.at<uchar>(i, j) =( img.at<uchar>(i-1, j-1)+ img.at<uchar>(i-1, j)+ img.at<uchar>(i-1, j+1)+
						img.at<uchar>(i, j-1)+ img.at<uchar>(i, j+1)+
						img.at<uchar>(i+1, j-1)+ img.at<uchar>(i+1, j)+ img.at<uchar>(i+1, j-1))/8;
				}
				else
				{
					img.at<uchar>(i, j) = 0;
				}
			}
		}*/
	imshow("wtf2", img);
	return img;

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


	minus(minm, ves);
	imshow("min", min);
	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

	min = vascularSeg(min);
	cv::subtract(min, ves, min);

	imshow("min dilate", min);

	//waitKey();
}

void Test1() {

	char fname[MAX_PATH];
	if (openFileDlg(fname)) {

		Mat src;
		src = imread(fname);

		waitKey();
	}
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
		//imshow("invlargeVessels", largeVessels);

		//imshow("out", src);
		//imshow("out2", src3);
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
		Mat contrasts = contrast(doamneAjuta,1.75); //inainte 2.0
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

		important= removeSmallComponentsNoise(thresholded,60);

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

		contrasts = removeSmallComponentsNoise(contrasts, 75);

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

		Classificare1(retine, important);

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
		}
	}
	while (op!=0);
	return 0;
}
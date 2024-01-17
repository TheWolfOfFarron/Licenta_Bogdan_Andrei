// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>

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

	waitKey();
}

void Test1() {

	char fname[MAX_PATH];
	if (openFileDlg(fname)) {

		Mat src;
		src = imread(fname);

	//	Mat dst = contextAwareSaliencyDetection(src);
	//	imshow("Dasd", dst);
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
	cv::medianBlur(src, enhanced, 7); // Apply median filter to remove noise
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	clahe->setClipLimit(4);
	clahe->apply(enhanced, enhanced); // Enhance contrast
	return enhanced;
}

// Function to extract vessels
cv::Mat extractVessels(const cv::Mat& src) {
	cv::Mat vessels;
	// Instead of Canny, we could try adaptive thresholding which might be better at picking up vessels
	cv::adaptiveThreshold(src, vessels, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
		cv::THRESH_BINARY_INV, 11, 2);
	return vessels;
}

// Function to remove small structures and keep large vessels
cv::Mat keepLargeStructures(const cv::Mat& src) {
	cv::Mat largeVessels;
	// Use morphological closing to keep large structures
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::morphologyEx(src, largeVessels, cv::MORPH_CLOSE, kernel);
	return largeVessels;
}


cv::Mat keepEdgeVessels(const cv::Mat& src) {
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(src.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat edgeVessels = cv::Mat::zeros(src.size(), CV_8UC1);

	// Loop over the contours
	for (size_t i = 0; i < contours.size(); i++) {
		// Approximate the contour to a polygon
		std::vector<cv::Point> polygon;
		cv::approxPolyDP(contours[i], polygon, 1.0, true);

		// Check if the contour starts from the edge of the image
		bool startsFromEdge = false;
		for (const cv::Point& pt : polygon) {
			if (pt.x == 0 || pt.y == 0 || pt.x == src.cols - 1 || pt.y == src.rows - 1) {
				startsFromEdge = true;
				break;
			}
		}

		// If it starts from the edge and is large enough, draw it on the result image
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

	// Loop over the contours to find edge vessels
	for (size_t i = 0; i < contours.size(); i++) {
		// Check if the contour starts from the edge of the image
		bool startsFromEdge = false;
		for (const cv::Point& pt : contours[i]) {
			if (pt.x <= 1 || pt.y <= 1 || pt.x >= src.cols - 2 || pt.y >= src.rows - 2) {
				startsFromEdge = true;
				break;
			}
		}

		// If it starts from the edge and is large enough, draw it on the result image
		if (startsFromEdge && cv::contourArea(contours[i]) > 100) { // Threshold for "large" is set to 100 here
			cv::drawContours(edgeVessels, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);
		}
	}

	// Dilate the result to include contours that are within 2 pixels of the edge vessels
	cv::Mat dilatedEdgeVessels;
	cv::dilate(edgeVessels, dilatedEdgeVessels, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));

	// Use the dilated image as a mask to include original contours that are close to the edge vessels
	cv::Mat finalVessels;
	src.copyTo(finalVessels, dilatedEdgeVessels);

	return finalVessels;
}



void chat() {
	char fname[MAX_PATH];
	if (openFileDlg(fname)) {


		cv::Mat retina= imread(fname);
		Mat src = retina.clone();
		Mat src3 = retina.clone();
		cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
		cv::cvtColor(src3, src3, cv::COLOR_BGR2GRAY);
		// Preprocess the image
		cv::Mat preprocessed = preprocess(retina);
		imshow("preprocessed", preprocessed);

		// Enhance vessels
		cv::Mat enhanced = enhanceVessels(preprocessed);
		imshow("enhanced", enhanced);

		// Extract vessels
		cv::Mat vessels = extractVessels(enhanced);
		imshow("vessels", vessels);

		// Keep only large structures
		cv::Mat largeVessels = keepLargeStructures(vessels);


		for(int i=0;i<src.rows;i++)
			for (int j = 0; j < src.cols; j++) {
				if (vessels.at<uchar>(i, j) == 0)
					src.at<uchar>(i, j) = 0;
			}

		openFileDlg(fname);
		 
		Mat src2 = imread(fname,IMREAD_GRAYSCALE);
		for (int i = 0; i < src2.rows; i++)
			for (int j = 0; j < src2.cols; j++) {
				if (src2.at<uchar>(i, j) == 255)
					src3.at<uchar>(i, j) = 0;
			}
		Mat l1 = keepEdgeVessels(largeVessels);

		imshow("largeVessels", largeVessels);
		for (int i = 0; i < largeVessels.rows; i++)
			for (int j = 0; j < largeVessels.cols; j++) {
				if (largeVessels.at<uchar>(i, j) == 255)
					largeVessels.at<uchar>(i, j) = 0;
				else
					largeVessels.at<uchar>(i, j) = 255;

			}
		imshow("invlargeVessels", largeVessels);

		imshow("out", src);
		imshow("out2", src3);
		Mat l= keepEdgeVesselsv2(largeVessels);
		imshow("inv", l);
		imshow("notinv", l1);
		waitKey();

	}
}


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
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
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative\n");
		printf(" 4 - Image negative (fast)\n");
		printf(" 5 - BGR->Gray\n");
		printf(" 6 - BGR->Gray (fast, save result to disk) \n");
		printf(" 7 - BGR->HSV\n");
		printf(" 8 - Resize image\n");
		printf(" 9 - Canny edge detection\n");
		printf(" 10 - Edges in a video sequence\n");
		printf(" 11 - Snap frame from live video\n");
		printf(" 12 - Mouse callback demo\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testNegativeImage();
				break;
			case 4:
				testNegativeImageFast();
				break;
			case 5:
				testColor2Gray();
				break;
			case 6:
				testImageOpenAndSave();
				break;
			case 7:
				testBGR2HSV();
				break;
			case 8:
				testResize();
				break;
			case 9:
				testCanny();
				break;
			case 10:
				testVideoSequence();
				break;
			case 11:
				testSnap();
				break;
			case 12:
				testMouseClick();
				break;
			case 13:
				char fname[MAX_PATH];
				if (openFileDlg(fname)) {
					Mat src = imread(fname, IMREAD_GRAYSCALE);
					imshow("src", src);
					prepoc(src);
				}
			case 14: 
				chat();
				break;
		}
	}
	while (op!=0);
	return 0;
}
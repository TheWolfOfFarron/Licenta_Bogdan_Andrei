// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>

wchar_t* projectPath;





// Function to calculate brightness contrast
double calculateBrightnessContrast(const Mat& patch) {
	// Calculate the mean intensity of the patch (excluding the central pixel)
	Scalar meanIntensity = mean(patch(Rect(1, 1, patch.cols - 2, patch.rows - 2)));

	// Get the intensity of the central pixel
	double centralIntensity = patch.at<uchar>(1, 1);

	// Calculate brightness contrast
	double brightnessContrast = std::abs(centralIntensity - meanIntensity[0]);

	// Normalize to the range [0, 1]
	brightnessContrast /= 255.0;

	return brightnessContrast;
}

// Function to calculate positional distance between two patches
double calculatePositionalDistance(const Rect& patch1Rect, const Rect& patch2Rect) {
	// Calculate the center of each patch
	Point2f center1(patch1Rect.x + patch1Rect.width / 2.0, patch1Rect.y + patch1Rect.height / 2.0);
	Point2f center2(patch2Rect.x + patch2Rect.width / 2.0, patch2Rect.y + patch2Rect.height / 2.0);

	// Calculate Euclidean distance between the centers
	double positionalDistance = norm(center1 - center2);

	// Normalize to the range [0, 1] based on the image size
	positionalDistance /= max(patch1Rect.width, patch1Rect.height);

	return positionalDistance;
}






Mat calculateOrientationContrast(const Mat& patch) {
	Mat orientationContrast;
	double brightnessContrast = calculateBrightnessContrast(patch);
	// Parameters for Gabor filter
	int kernelSize = 31;  // Adjust as needed
	double sigma = 5.0;   // Adjust as needed
	double theta[4] = { 0, CV_PI / 4, CV_PI / 2, 3 * CV_PI / 4 };

	// Initialize the result matrix
	orientationContrast = Mat::zeros(patch.size(), CV_64F);

	// Apply Gabor filter at four different orientations
	for (int i = 0; i < 4; ++i) {
		// Create Gabor kernel
		Mat gaborKernel = getGaborKernel(Size(kernelSize, kernelSize), sigma, theta[i], 10.0, 0.5, 0, CV_64F);

		// Apply filter to the patch
		Mat filteredImage;
		filter2D(patch, filteredImage, CV_64F, gaborKernel);

		// Accumulate squared response for orientation contrast
		multiply(filteredImage, filteredImage, filteredImage);  // Element-wise multiplication
		orientationContrast += filteredImage;
	}

	// Normalize the orientationContrast matrix to the range [0, 1]
	normalize(orientationContrast, orientationContrast, 0, 1, NORM_MINMAX);


	Rect patchRect(0, 0, patch.cols, patch.rows);  // Assuming the input patch is the entire patch
	double positionalDistance = calculatePositionalDistance(patchRect, patchRect);

	Mat distinctiveness = brightnessContrast * orientationContrast;

	return orientationContrast;
}

// Function to calculate distinctiveness measure using Eq. (2)
double calculateDistinctiveness(const double brightnessContrast, const Mat& orientationContrast, double positionalDistance) {
	// TODO: Implement Eq. (2) to calculate distinctiveness measure
	double c = 3.0; // You may adjust the value of c

	// Calculate distinctiveness using the provided formula
	Mat distinctiveness = (brightnessContrast + orientationContrast) / (2 * (1 + c * positionalDistance));



	return sum(distinctiveness)[0];
}





Mat multiScaleSaliencyDetection(const Mat& inputImage, int K) {
	Mat saliencyMap;

	// Parameters for multi-scale saliency detection
	std::vector<double> scales = { 1.0, 0.8, 0.5, 0.3 };  // Adjust as needed
	int patchSize = 7;  // Adjust as needed
	int overlap = 50;   // Percentage overlap
	int numPatches = K;

	// Initialize the saliency map
	saliencyMap = Mat::zeros(inputImage.size(), CV_64F);

	// Loop over each scale
	for (double scale : scales) {
		// Resize the image
		Mat resizedImage;
		resize(inputImage, resizedImage, Size(), scale, scale);

		// Calculate the saliency map at the current scale
		for (int i = 0; i < numPatches; ++i) {
			int x = rand() % (resizedImage.cols - patchSize + 1);
			int y = rand() % (resizedImage.rows - patchSize + 1);

			// Extract the patch
			Rect patchRect(x, y, patchSize, patchSize);
			Mat patch = resizedImage(patchRect);

			// TODO: Calculate brightness contrast
			double brightnessContrast = calculateBrightnessContrast(patch);
			std::cout << "Debug Message: This point is reached 1." << std::endl;
			// TODO: Calculate orientation contrast
			Mat orientationContrast = calculateOrientationContrast(patch);
			std::cout << "Debug Message: This point is reached 2." << std::endl;
			// TODO: Calculate positional distance
			double positionalDistance = calculatePositionalDistance(patchRect, patchRect);
			std::cout << "Debug Message: This point is reached 3." << std::endl;
			// TODO: Calculate distinctiveness using Eq. (2)
			double distinctiveness = calculateDistinctiveness(brightnessContrast, orientationContrast, positionalDistance);
			std::cout << "Debug Message: This point is reached 4." << std::endl;
			// TODO: Calculate saliency value at pixel (x, y) using Eq. (4)
			double saliencyValue = 1.0 / (1.0 + exp(-distinctiveness));
			std::cout << "Debug Message: This point is reached 5." << std::endl;

			// Update the saliency map
			saliencyMap(Rect(x / scale, y / scale, patchSize / scale, patchSize / scale)) += saliencyValue;
			std::cout << "Debug Message: This point is reached 6." << std::endl;

		}
		std::cout << "Debug Message: This point is reached 6.//////" << scale << std::endl;
		break;
	}

	// Normalize the saliency map to the range [0, 1]
	normalize(saliencyMap, saliencyMap, 0, 1, NORM_MINMAX);

	return saliencyMap;
}

// Adjust saliency values based on distance to foci using Eq. (6)
Mat adjustSaliencyByDistance(const Mat& saliencyMap, double distanceToFocus) {
	Mat adjustedSaliency;

	// Parameters for Eq. (6)
	double threshold = 0.8; // You may adjust the threshold value

	// Apply threshold to identify the most attended localized areas
	Mat attendedAreas = (saliencyMap > threshold);

	// Calculate the Euclidean distance to the closest attended pixel
	distanceTransform(attendedAreas, adjustedSaliency, DIST_L2, DIST_MASK_PRECISE);

	// Normalize adjustedSaliency to the range [0, 1]
	normalize(adjustedSaliency, adjustedSaliency, 0, 1, NORM_MINMAX);

	// Apply Eq. (6) to adjust saliency values
	pow(saliencyMap, distanceToFocus, adjustedSaliency);
	adjustedSaliency = adjustedSaliency.mul(1.0 - distanceToFocus);

	return adjustedSaliency;
}



Mat contextAwareSaliencyDetection(const Mat& inputImage) {
	Mat saliencyMap;

	// Convert input image to grayscale
	Mat grayImage;
	cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);

	// Define parameters
	int patchSize = 7;
	std::vector<double> scales = { 1.0, 0.8, 0.5, 0.3 };
	double focalThreshold = 0.8;
	std::cout << "Dasda" << '\n';
	// Iterate over scales
	for (double scale : scales) {
		// Resize the image
		Mat resizedImage;
		resize(grayImage, resizedImage, Size(), scale, scale);
		std::cout << "Dasda" << '\n';
		// Iterate over pixels
		for (int i = patchSize / 2; i < resizedImage.rows - patchSize / 2; ++i) {
			for (int j = patchSize / 2; j < resizedImage.cols - patchSize / 2; ++j) {
				// Extract the patch centered at (i, j)
				std::cout << "Dasda" << '\n';
				Rect patchRect(j - patchSize / 2, i - patchSize / 2, patchSize, patchSize);
				std::cout << "Dasda" << '\n';
				Mat patch = resizedImage(patchRect);
				std::cout << "Dasda" << '\n';

				// Calculate brightness contrast
				double brightnessContrast = calculateBrightnessContrast(patch);
				std::cout << "Dasda" << '\n';
				// Calculate orientation contrast using Gabor filters
				Mat orientationContrast = calculateOrientationContrast(patch);
				std::cout << "Debug Message: This point is reached." << std::endl;
				// Calculate positional distance
				double positionalDistance = calculatePositionalDistance(patchRect, patchRect);
				std::cout << "Debug Message: This point is reached." << std::endl;
				// Calculate distinctiveness measure using Eq. (2)
				double distinctiveness = calculateDistinctiveness(brightnessContrast, orientationContrast, positionalDistance);
				std::cout << "Debug Message: This point is reached." << std::endl;
				std::cout << "#####################################################################" << std::endl;
				// Perform multi-scale saliency detection using Eq. (4)
				Mat multiScaleSaliency = multiScaleSaliencyDetection(patch, 5);  // You may adjust the parameter K
				std::cout << "Debug Message: This point is reached." << std::endl;
				// Adjust saliency values based on distance to foci using Eq. (6)
				Mat adjustedSaliency = adjustSaliencyByDistance(multiScaleSaliency, positionalDistance);
				std::cout << "Debug Message: This point is reached." << std::endl;
				// Accumulate saliency values
				saliencyMap += adjustedSaliency;
			}
		}
	}

	// Normalize the saliency map to the range [0, 1]
	normalize(saliencyMap, saliencyMap, 0, 255, NORM_MINMAX);
	std::cout << "Debug Message: This point is reached." << std::endl;

	// Apply threshold to extract the most attended localized areas
	cv::threshold(saliencyMap, saliencyMap, focalThreshold, 1.0, THRESH_BINARY);

	return saliencyMap;
}



void Test1() {

	char fname[MAX_PATH];
	if (openFileDlg(fname)) {

		Mat src;
		src = imread(fname);

		Mat dst = contextAwareSaliencyDetection(src);
		imshow("Dasd", dst);
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
		}
	}
	while (op!=0);
	return 0;
}
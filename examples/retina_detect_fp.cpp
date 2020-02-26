#include <stdio.h>
#include <algorithm>
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include <cmath>

using namespace cv;
using namespace std;
using namespace caffe;

struct anchor_win
{
	float x_ctr;
	float y_ctr;
	float w;
	float h;
};

struct anchor_box
{
	float x1;
	float y1;
	float x2;
	float y2;
};

struct FacePts
{
	float x[5];
	float y[5];
};

struct FaceDetectInfo
{
	float score;
	anchor_box rect;
	FacePts pts;
};

struct anchor_cfg
{
public:
	int STRIDE;
	vector<int> SCALES;
	int BASE_SIZE;
	vector<float> RATIOS;
	int ALLOWED_BORDER;

	anchor_cfg()
	{
		STRIDE = 0;
		SCALES.clear();
		BASE_SIZE = 0;
		RATIOS.clear();
		ALLOWED_BORDER = 0;
	}
};

//processing
anchor_win  _whctrs(anchor_box anchor)
{
	//Return width, height, x center, and y center for an anchor (window).
	anchor_win win;
	win.w = anchor.x2 - anchor.x1 + 1;
	win.h = anchor.y2 - anchor.y1 + 1;
	win.x_ctr = anchor.x1 + 0.5 * (win.w - 1);
	win.y_ctr = anchor.y1 + 0.5 * (win.h - 1);

	return win;
}

anchor_box _mkanchors(anchor_win win)
{
	//Given a vector of widths (ws) and heights (hs) around a center
	//(x_ctr, y_ctr), output a set of anchors (windows).
	anchor_box anchor;
	anchor.x1 = win.x_ctr - 0.5 * (win.w - 1);
	anchor.y1 = win.y_ctr - 0.5 * (win.h - 1);
	anchor.x2 = win.x_ctr + 0.5 * (win.w - 1);
	anchor.y2 = win.y_ctr + 0.5 * (win.h - 1);

	return anchor;
}

vector<anchor_box> _ratio_enum(anchor_box anchor, vector<float> ratios)
{
	//Enumerate a set of anchors for each aspect ratio wrt an anchor.
	vector<anchor_box> anchors;
	for (size_t i = 0; i < ratios.size(); i++) {
		anchor_win win = _whctrs(anchor);
		float size = win.w * win.h;
		float scale = size / ratios[i];

		win.w = std::round(sqrt(scale));
		win.h = std::round(win.w * ratios[i]);

		anchor_box tmp = _mkanchors(win);
		anchors.push_back(tmp);
	}

	return anchors;
}

vector<anchor_box> _scale_enum(anchor_box anchor, vector<int> scales)
{
	//Enumerate a set of anchors for each scale wrt an anchor.
	vector<anchor_box> anchors;
	for (size_t i = 0; i < scales.size(); i++) {
		anchor_win win = _whctrs(anchor);

		win.w = win.w * scales[i];
		win.h = win.h * scales[i];

		anchor_box tmp = _mkanchors(win);
		anchors.push_back(tmp);
	}

	return anchors;
}

vector<anchor_box> generate_anchors(int base_size = 16, vector<float> ratios = { 0.5, 1, 2 },
	vector<int> scales = { 8, 64 }, int stride = 16, bool dense_anchor = false)
{
	//Generate anchor (reference) windows by enumerating aspect ratios X
	//scales wrt a reference (0, 0, 15, 15) window.

	anchor_box base_anchor;
	base_anchor.x1 = 0;
	base_anchor.y1 = 0;
	base_anchor.x2 = base_size - 1;
	base_anchor.y2 = base_size - 1;

	vector<anchor_box> ratio_anchors;
	ratio_anchors = _ratio_enum(base_anchor, ratios);

	vector<anchor_box> anchors;
	for (size_t i = 0; i < ratio_anchors.size(); i++) {
		vector<anchor_box> tmp = _scale_enum(ratio_anchors[i], scales);
		anchors.insert(anchors.end(), tmp.begin(), tmp.end());
	}

	if (dense_anchor) {
		assert(stride % 2 == 0);
		vector<anchor_box> anchors2 = anchors;
		for (size_t i = 0; i < anchors2.size(); i++) {
			anchors2[i].x1 += stride / 2;
			anchors2[i].y1 += stride / 2;
			anchors2[i].x2 += stride / 2;
			anchors2[i].y2 += stride / 2;
		}
		anchors.insert(anchors.end(), anchors2.begin(), anchors2.end());
	}

	return anchors;
}

vector<vector<anchor_box> > generate_anchors_fpn(bool dense_anchor = false, vector<anchor_cfg> cfg = {})
{
	//Generate anchor (reference) windows by enumerating aspect ratios X
	//scales wrt a reference (0, 0, 15, 15) window.

	vector<vector<anchor_box> > anchors;
	for (size_t i = 0; i < cfg.size(); i++) {
		//stride从小到大[32 16 8]
		anchor_cfg tmp = cfg[i];
		int bs = tmp.BASE_SIZE;
		vector<float> ratios = tmp.RATIOS;
		vector<int> scales = tmp.SCALES;
		int stride = tmp.STRIDE;

		vector<anchor_box> r = generate_anchors(bs, ratios, scales, stride, dense_anchor);
		anchors.push_back(r);
	}

	return anchors;
}

vector<anchor_box> anchors_plane(int height, int width, int stride, vector<anchor_box> base_anchors)
{
	/*
	height: height of plane
	width:  width of plane
	stride: stride ot the original image
	anchors_base: a base set of anchors
	*/

	vector<anchor_box> all_anchors;
	for (size_t k = 0; k < base_anchors.size(); k++) {
		for (int ih = 0; ih < height; ih++) {
			int sh = ih * stride;
			for (int iw = 0; iw < width; iw++) {
				int sw = iw * stride;

				anchor_box tmp;
				tmp.x1 = base_anchors[k].x1 + sw;
				tmp.y1 = base_anchors[k].y1 + sh;
				tmp.x2 = base_anchors[k].x2 + sw;
				tmp.y2 = base_anchors[k].y2 + sh;
				all_anchors.push_back(tmp);
			}
		}
	}

	return all_anchors;
}

void clip_boxes(vector<anchor_box> & boxes, int width, int height)
{
	//Clip boxes to image boundaries.
	for (size_t i = 0; i < boxes.size(); i++) {
		if (boxes[i].x1 < 0) {
			boxes[i].x1 = 0;
		}
		if (boxes[i].y1 < 0) {
			boxes[i].y1 = 0;
		}
		if (boxes[i].x2 > width - 1) {
			boxes[i].x2 = width - 1;
		}
		if (boxes[i].y2 > height - 1) {
			boxes[i].y2 = height - 1;
		}
		//        boxes[i].x1 = std::max<float>(std::min<float>(boxes[i].x1, width - 1), 0);
		//        boxes[i].y1 = std::max<float>(std::min<float>(boxes[i].y1, height - 1), 0);
		//        boxes[i].x2 = std::max<float>(std::min<float>(boxes[i].x2, width - 1), 0);
		//        boxes[i].y2 = std::max<float>(std::min<float>(boxes[i].y2, height - 1), 0);
	}
}

void clip_boxes(anchor_box & box, int width, int height)
{
	//Clip boxes to image boundaries.
	if (box.x1 < 0) {
		box.x1 = 0;
	}
	if (box.y1 < 0) {
		box.y1 = 0;
	}
	if (box.x2 > width - 1) {
		box.x2 = width - 1;
	}
	if (box.y2 > height - 1) {
		box.y2 = height - 1;
	}
	//    boxes[i].x1 = std::max<float>(std::min<float>(boxes[i].x1, width - 1), 0);
	//    boxes[i].y1 = std::max<float>(std::min<float>(boxes[i].y1, height - 1), 0);
	//    boxes[i].x2 = std::max<float>(std::min<float>(boxes[i].x2, width - 1), 0);
	//    boxes[i].y2 = std::max<float>(std::min<float>(boxes[i].y2, height - 1), 0);

}



vector<anchor_box> bbox_pred(vector<anchor_box> anchors, vector<cv::Vec4f> regress)
{
	//"""
	//  Transform the set of class-agnostic boxes into class-specific boxes
	//  by applying the predicted offsets (box_deltas)
	//  :param boxes: !important [N 4]
	//  :param box_deltas: [N, 4 * num_classes]
	//  :return: [N 4 * num_classes]
	//  """

	vector<anchor_box> rects(anchors.size());
	for (size_t i = 0; i < anchors.size(); i++) {
		float width = anchors[i].x2 - anchors[i].x1 + 1;
		float height = anchors[i].y2 - anchors[i].y1 + 1;
		float ctr_x = anchors[i].x1 + 0.5 * (width - 1.0);
		float ctr_y = anchors[i].y1 + 0.5 * (height - 1.0);

		float pred_ctr_x = regress[i][0] * width + ctr_x;
		float pred_ctr_y = regress[i][1] * height + ctr_y;
		float pred_w = exp(regress[i][2]) * width;
		float pred_h = exp(regress[i][3]) * height;

		rects[i].x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
		rects[i].y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
		rects[i].x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
		rects[i].y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);
	}

	return rects;
}

anchor_box bbox_pred(anchor_box anchor, cv::Vec4f regress)
{
	anchor_box rect;

	float width = anchor.x2 - anchor.x1 + 1;
	float height = anchor.y2 - anchor.y1 + 1;
	float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
	float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

	float pred_ctr_x = regress[0] * width + ctr_x;
	float pred_ctr_y = regress[1] * height + ctr_y;
	float pred_w = exp(regress[2]) * width;
	float pred_h = exp(regress[3]) * height;

	rect.x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
	rect.y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
	rect.x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
	rect.y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);

	return rect;
}

vector<FacePts> landmark_pred(vector<anchor_box> anchors, vector<FacePts> facePts)
{
	vector<FacePts> pts(anchors.size());
	for (size_t i = 0; i < anchors.size(); i++) {
		float width = anchors[i].x2 - anchors[i].x1 + 1;
		float height = anchors[i].y2 - anchors[i].y1 + 1;
		float ctr_x = anchors[i].x1 + 0.5 * (width - 1.0);
		float ctr_y = anchors[i].y1 + 0.5 * (height - 1.0);

		for (size_t j = 0; j < 5; j++) {
			pts[i].x[j] = facePts[i].x[j] * width + ctr_x;
			pts[i].y[j] = facePts[i].y[j] * height + ctr_y;
		}
	}

	return pts;
}

FacePts landmark_pred(anchor_box anchor, FacePts facePt)
{
	FacePts pt;
	float width = anchor.x2 - anchor.x1 + 1;
	float height = anchor.y2 - anchor.y1 + 1;
	float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
	float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

	for (size_t j = 0; j < 5; j++) {
		pt.x[j] = facePt.x[j] * width + ctr_x;
		pt.y[j] = facePt.y[j] * height + ctr_y;
	}

	return pt;
}

bool CompareBBox(const FaceDetectInfo & a, const FaceDetectInfo & b)
{
	return a.score > b.score;
}

std::vector<FaceDetectInfo> nms(std::vector<FaceDetectInfo> & bboxes, float threshold)
{
	std::vector<FaceDetectInfo> bboxes_nms;
	std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

	int32_t select_idx = 0;
	int32_t num_bbox = static_cast<int32_t>(bboxes.size());
	std::vector<int32_t> mask_merged(num_bbox, 0);
	bool all_merged = false;

	while (!all_merged) {
		while (select_idx < num_bbox && mask_merged[select_idx] == 1)
			select_idx++;
		if (select_idx == num_bbox) {
			all_merged = true;
			continue;
		}

		bboxes_nms.push_back(bboxes[select_idx]);
		mask_merged[select_idx] = 1;

		anchor_box select_bbox = bboxes[select_idx].rect;
		float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) * (select_bbox.y2 - select_bbox.y1 + 1));
		float x1 = static_cast<float>(select_bbox.x1);
		float y1 = static_cast<float>(select_bbox.y1);
		float x2 = static_cast<float>(select_bbox.x2);
		float y2 = static_cast<float>(select_bbox.y2);

		select_idx++;
		for (int32_t i = select_idx; i < num_bbox; i++) {
			if (mask_merged[i] == 1)
				continue;

			anchor_box & bbox_i = bboxes[i].rect;
			float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
			float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
			float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;   //<- float 型不加1
			float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
			if (w <= 0 || h <= 0)
				continue;

			float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) * (bbox_i.y2 - bbox_i.y1 + 1));
			float area_intersect = w * h;


			if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > threshold) {
				mask_merged[i] = 1;
			}
		}
	}

	return bboxes_nms;
}

int main(void)
{
	///////////////////////for caffe//////////////////////////////
	Caffe::set_mode(Caffe::CPU);
	boost::shared_ptr<Net<float> > Net_;
	//Net_.reset(new Net<float>(("/home/abab2365/workspace/etri1/int8_models/mnet/model/converted_deploy.prototxt"), TEST));
	//Net_->CopyTrainedLayersFrom(("/home/abab2365/workspace/etri1/int8_models/mnet/model/converted_model.caffemodel"));
	Net_.reset(new Net<float>(("/home/abab2365/workspace/etri1/int8_models/mnet/model/new_deploy.prototxt"), TEST));
	Net_->CopyTrainedLayersFrom(("/home/abab2365/workspace/etri1/int8_models/mnet/model/new_model.caffemodel"));

	///////////////////////for caffe//////////////////////////////
	cv::Mat img;
	img = imread("/home/abab2365/t2.jpg", IMREAD_COLOR);
	float threshold= 0.85;
	float scales;

	float nms_threshold = 0.5;
	vector<int> _feat_stride_fpn;
	map<string, int> _num_anchors;
	bool dense_anchor = false;
	anchor_cfg tmp;
	vector<anchor_cfg> cfg;
	vector<float> _ratio;
	map<string, vector<anchor_box> > _anchors_fpn;
	_feat_stride_fpn = { 32, 16, 8 };
	_ratio = { 1.0 };
	tmp.SCALES = { 32, 16 };
	tmp.BASE_SIZE = 16;
	tmp.RATIOS = _ratio;
	tmp.ALLOWED_BORDER = 9999;
	tmp.STRIDE = 32;
	cfg.push_back(tmp);

	tmp.SCALES = { 8, 4 };
	tmp.BASE_SIZE = 16;
	tmp.RATIOS = _ratio;
	tmp.ALLOWED_BORDER = 9999;
	tmp.STRIDE = 16;
	cfg.push_back(tmp);

	tmp.SCALES = { 2, 1 };
	tmp.BASE_SIZE = 16;
	tmp.RATIOS = _ratio;
	tmp.ALLOWED_BORDER = 9999;
	tmp.STRIDE = 8;
	cfg.push_back(tmp);
	vector<vector<anchor_box> > anchors_fpn = generate_anchors_fpn(dense_anchor, cfg);
	for (size_t i = 0; i < anchors_fpn.size(); i++) {
		string key = "stride" + std::to_string(_feat_stride_fpn[i]);
		_anchors_fpn[key] = anchors_fpn[i];
		_num_anchors[key] = anchors_fpn[i].size();
	}
	if (img.empty()) {
		return -1;
	}

	//////////////////////////////////////start preprocess/////////////////////////////////////////////////
	//double pre = (double)getTickCount();
	int ws = (img.cols + 31) / 32 * 32;
	int hs = (img.rows + 31) / 32 * 32;

	cv::copyMakeBorder(img, img, 0, hs - img.rows, 0, ws - img.cols, cv::BORDER_CONSTANT, cv::Scalar(0));

	cv::Mat src = img.clone();

	//to float
	img.convertTo(img, CV_32FC3);

	//rgb
	cvtColor(img, img, CV_BGR2RGB);

	Blob<float> * input_layer = Net_->input_blobs()[0];

	input_layer->Reshape(1, 3, img.rows, img.cols);
	Net_->Reshape();

	vector<Mat> input_channels;

	int width = input_layer->width();
	int height = input_layer->height();
	
	printf("layer width = %d\n", width);
	printf("layer height = %d\n", height);

	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		Mat channel(height, width, CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += width * height;
	}

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the Mat
	* objects in input_channels. */
	cv::split(img, input_channels);
	//cv::imwrite("./preprocess_done_img.jpg", img);
	//pre = (double)getTickCount() - pre;
	//std::cout << "pre compute time :" << pre * 1000.0 / cv::getTickFrequency() << " ms \n";
	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////LOG(INFO) << "Start net_->Forward()";///////////////////////////////
	//double t1 = (double)getTickCount();
	Net_->Forward();	//inference
	//t1 = (double)getTickCount() - t1;
	//std::cout << "infer compute time :" << t1 * 1000.0 / cv::getTickFrequency() << " ms \n";
	/////////////////////////////////////LOG(INFO) << "Done net_->Forward()";////////////////////////////////
	/////////////////////////////////////postprocess/////////////////////////////////////////////////////////
	//double post = (double)getTickCount();
	string name_bbox = "face_rpn_bbox_pred_";
	string name_score = "face_rpn_cls_prob_reshape_";
	string name_landmark = "face_rpn_landmark_pred_";

	vector<FaceDetectInfo> faceInfo;
	for (size_t i = 0; i < _feat_stride_fpn.size(); i++) {
		///////////////////////////////////////////////
				//double s1 = (double)getTickCount();
		///////////////////////////////////////////////
		string key = "stride" + std::to_string(_feat_stride_fpn[i]);
		int stride = _feat_stride_fpn[i];

		string str = name_score + key;
		const boost::shared_ptr<Blob<float> > score_blob = Net_->blob_by_name(str);
		const float* scoreB = score_blob->cpu_data() + score_blob->count() / 2;
		const float* scoreE = scoreB + score_blob->count() / 2;
		std::vector<float> score = std::vector<float>(scoreB, scoreE);

		printf("score %d\n", i);
		for(int i=0;i<score.size();i++) {
			printf("%lf ", score[i]);
		}
		printf("\n");

		str = name_bbox + key;
		const boost::shared_ptr<Blob<float> > bbox_blob = Net_->blob_by_name(str);
		const float* bboxB = bbox_blob->cpu_data();
		const float* bboxE = bboxB + bbox_blob->count();
		std::vector<float> bbox_delta = std::vector<float>(bboxB, bboxE);

		printf("bbox_delta %d\n", i);
		for(int i=0;i<bbox_delta.size();i++) {
			printf("%lf ", bbox_delta[i]);
		}
		printf("\n");

		str = name_landmark + key;
		const boost::shared_ptr<Blob<float> > landmark_blob = Net_->blob_by_name(str);
		const float* landmarkB = landmark_blob->cpu_data();
		const float* landmarkE = landmarkB + landmark_blob->count();
		std::vector<float> landmark_delta = std::vector<float>(landmarkB, landmarkE);

		printf("landmark_delta %d\n", i);
		for(int i=0;i<landmark_delta.size();i++) {
			printf("%lf ", landmark_delta[i]);
		}
		printf("\n");

		int width = score_blob->width();
		int height = score_blob->height();
		size_t count = width * height;
		size_t num_anchor = _num_anchors[key];

		///////////////////////////////////////////////
				//s1 = (double)getTickCount() - s1;
				//std::cout << "s1 compute time :" << s1*1000.0 / cv::getTickFrequency() << " ms \n";
		///////////////////////////////////////////////

		vector<anchor_box> anchors = anchors_plane(height, width, stride, _anchors_fpn[key]);

		for (size_t num = 0; num < num_anchor; num++) {
			for (size_t j = 0; j < count; j++) {
				float conf = score[j + count * num];
				printf("confidence = %f % \n", 100*conf);
				if (conf <= threshold) {
					continue;
				}

				cv::Vec4f regress;
				float dx = bbox_delta[j + count * (0 + num * 4)];
				float dy = bbox_delta[j + count * (1 + num * 4)];
				float dw = bbox_delta[j + count * (2 + num * 4)];
				float dh = bbox_delta[j + count * (3 + num * 4)];
				regress = cv::Vec4f(dx, dy, dw, dh);


				anchor_box rect = bbox_pred(anchors[j + count * num], regress);

				clip_boxes(rect, ws, hs);

				FacePts pts;
				for (size_t k = 0; k < 5; k++) {
					pts.x[k] = landmark_delta[j + count * (num * 10 + k * 2)];
					pts.y[k] = landmark_delta[j + count * (num * 10 + k * 2 + 1)];
				}

				FacePts landmarks = landmark_pred(anchors[j + count * num], pts);

				FaceDetectInfo tmp;
				tmp.score = conf;
				tmp.rect = rect;
				tmp.pts = landmarks;
				faceInfo.push_back(tmp);
			}
		}
	}


	faceInfo = nms(faceInfo, nms_threshold);

	/*
	for(int u = 0; u < faceInfo.size(); u++) {
		cout << "face bbox xmin: " << faceInfo[u].rect.x1 << endl;
		cout << "face bbox ymin: " << faceInfo[u].rect.y1 << endl;
		cout << "face bbox xmax: " << faceInfo[u].rect.x2 << endl;
		cout << "face bbox ymax: " << faceInfo[u].rect.y2 << endl;
		for(int v = 0; v < 5; v++) {
			cout << "landmark x : " << faceInfo[u].pts.x[v] << endl;
			cout << "landmark y : " << faceInfo[u].pts.y[v] << endl;
		}

		printf("\n");
	}
	*/
	
	for(int u = 0; u < faceInfo.size(); u++) {
		cv::rectangle(img, cv::Point(faceInfo[u].rect.x1, faceInfo[u].rect.y1), cv::Point(faceInfo[u].rect.x2, faceInfo[u].rect.y2), cv::Scalar(0, 0, 255), 3);
		for(int v = 0; v < 5; v++) {
		//	cout << "landmark x : " << faceInfo[u].pts.x[v] << endl;
		//	cout << "landmark y : " << faceInfo[u].pts.y[v] << endl;
			cv::line(img, cv::Point(faceInfo[u].pts.x[v], faceInfo[u].pts.y[v]), cv::Point(faceInfo[u].pts.x[v], faceInfo[u].pts.y[v]), cv::Scalar(0, 0, 255), 3);
		}
	}
	
	//cv::imshow("result", img);
	cvtColor(img, img, CV_RGB2BGR);
	cv::imwrite("./result_img.jpg", img);

	cout << "face num: " << faceInfo.size() << endl;
	/*
	post = (double)getTickCount() - post;
	std::cout << "post compute time :" << post*1000.0 / cv::getTickFrequency() << " ms \n";


	for(size_t i = 0; i < faceInfo.size(); i++) {
		cv::Rect rect = cv::Rect(cv::Point2f(faceInfo[i].rect.x1, faceInfo[i].rect.y1), cv::Point2f(faceInfo[i].rect.x2, faceInfo[i].rect.y2));
		cv::rectangle(src, rect, Scalar(0, 0, 255), 2);

		for(size_t j = 0; j < 5; j++) {
			cv::Point2f pt = cv::Point2f(faceInfo[i].pts.x[j], faceInfo[i].pts.y[j]);
			cv::circle(src, pt, 1, Scalar(0, 255, 0), 2);
		}
	}

	imshow("dst", src);
	waitKey(0);
	*/
	return 0;
}

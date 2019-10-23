#include <stdio.h>
#include <cstring> 
#include <fstream> 
#include <iostream> 
#include <string> 
#include <vector> 
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include "boost/algorithm/string.hpp" 
#include "boost/filesystem.hpp" 
#include "boost/foreach.hpp" 

#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

using caffe::Caffe;

using namespace caffe;

void WrapInputLayer(std::vector<cv::Mat>* input_channels, Blob<float>* input_layer) {
  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}
/*
void image_crop(cv::Mat& old, cv::Mat& post, int height, int width) {
	for(int i=0;i<height;i++){
		for(int j=0;j<width;j++){
			post.data_at(i,j) = 
		}
	}
}
*/
void Preprocess(Net<float>& caffe_net, const cv::Mat& img, cv::Mat& mean,
                            std::vector<cv::Mat>* input_channels, int num_channels_, int height, int width) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample.convertTo(sample_float, CV_32FC3);
  else
    sample.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_resized;
  cv::Size input_geometry_(height, width);
  if (sample_float.size() != input_geometry_){
    cv::resize(sample_float, sample_resized, input_geometry_);
//	image_crop(smaple_float, sample_resized, height, width);
  }
  else
    sample_resized = sample_float;


  cv::Mat sample_normalized;
  cv::subtract(sample_resized, mean, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == caffe_net.input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int truth[50000];
int matchlabel[1200];

int main(int argc, char** argv)
{
	if(argc!=3){
		std::cout<<"parameter error\n";
	}
	std::string deploy(argv[1]);
	std::string model(argv[2]);
	std::string mean_name("/home/abab2365/workspace/etri1/caffe/data/ilsvrc12/dataset/imagenet_mean.binaryproto");
	std::string file_name("/home/abab2365/workspace/etri1/caffe/data/ilsvrc12/dataset/ilsvrc12_val_images/ILSVRC2012_val_");
	//std::string file_name("/home/abab2365/workspace/data/images/test/ILSVRC2012_val_");
	int num_images = 50000;

	Caffe::set_mode(Caffe::CPU);
	Net<float> caffe_net(deploy, caffe::TEST);
	caffe_net.CopyTrainedLayersFrom(model);

	// Set mean image mat
	std::cout<<"set mean image mat..\n";
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_name, &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	int num_channels_ = caffe_net.input_blobs()[0]->channels();
	CHECK_EQ(mean_blob.channels(), num_channels_)
	<< "Number of channels of mean file doesn't match input layer.";
	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}
	/* Merge the separate channels into a single image. */
	cv::Mat mean_;
	cv::merge(channels, mean_);
	/* Compute the global mean pixel value and create a mean image
	* filled with this value. */
	cv::Scalar channel_mean = cv::mean(mean_);
	cv::Size geometry(caffe_net.input_blobs()[0]->height(), caffe_net.input_blobs()[0]->width());
	cv::Mat mean(geometry, mean_.type(), channel_mean);

	FILE *match_file = fopen("/home/abab2365/workspace/_etc/mat2py/match_label.txt", "r");
	for(int i=1;i<=1000;i++){
		int a, b;
		fscanf(match_file, "%d %d", &a, &b);
		matchlabel[b] = a;
	}
	FILE *in = fopen("/home/abab2365/workspace/etri1/caffe/data/ilsvrc12/dataset/ILSVRC2012_validation_ground_truth.txt", "r");
	for(int i=1;i<49997;i++){
		fscanf(in, "%d", &truth[i]);
		truth[i] = matchlabel[truth[i]];
	}
	fclose(in);

/*	FILE *in = fopen("/home/abab2365/workspace/data/images/label.txt", "r");
	for(int i=1;i<49997;i++){
		fscanf(in, "%d", &truth[i]);
	}
	fclose(in);
*/	std::cout<<"Test inference\n";
	int image_count = 0;
	int predict_count = 0;
	for(int i=1;i<num_images;i++){
		if(truth[i]==-1) continue; // validatoin label is vaild when truth != -1
		char imsg[200];
		std::string imagepath = file_name;
		sprintf(imsg, "%08d.JPEG", i);
		imsg[strlen(imsg)]='\0';
		imagepath = imagepath + imsg;
		cv::Mat img = cv::imread(imagepath, -1);
		if(img.size().height==0){ //is if image size==0
			continue;
		}
		
		Blob<float>* input_layer = caffe_net.input_blobs()[0];
		std::vector<cv::Mat> input_channels;
		WrapInputLayer(&input_channels, input_layer);
    	Preprocess(caffe_net, img, mean, &input_channels, input_layer->channels(), input_layer->height(), input_layer->width());

    	caffe_net.Forward();

		Blob<float>* output_layer = caffe_net.output_blobs()[0];
		const float* begin = output_layer->cpu_data();
		const float* end = begin + output_layer->channels();
		std::vector<float> out(begin, end);
		int max = 0;
		for(int j=0;j<out.size();j++){
			//std::cout<<out[j]<<" ";
			if(out[j] > out[max]) max = j;
		}
		std::cout<<"\n"<<max<<" "<<out[max]<<"\n";

		int index = max;
		if(index==truth[i]){
			predict_count++;
		}
		image_count++;
		printf("predict rate: %lf\n", (double)predict_count / image_count);

		if(image_count%100==0){
			printf("%d: predict rate: %lf\n", image_count, (double)predict_count / image_count);
		}
	}

	return 0;
}


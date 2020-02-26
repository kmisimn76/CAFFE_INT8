#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#ifdef USE_OPENCV
using std::ofstream;
using caffe::Caffe;

using namespace caffe;

namespace fs = boost::filesystem;

#define INTERVAL_NUM 2048	//Set interval num of distribution

// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
  int count = 0;
#ifndef CPU_ONLY
  CUDA_CHECK(cudaGetDeviceCount(&count));
#else
  NO_GPU;
#endif
  for (int i = 0; i < count; ++i) {
    gpus->push_back(i);
  }
}


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

  int off = 0; //mobilenet
  cv::Mat sample_clip;// = sample_float;
  int col = sample_float.cols;
  int row = sample_float.rows;
  if (row < col) {
	  off = (col - row) / 2;
  	  cv::Rect rect(off, 0, row, row);
	  sample_clip = sample_float(rect);
  }
  else {
	  off = (row - col) / 2;
  	  cv::Rect rect(0, off, col, col);
	  sample_clip = sample_float(rect);
  }

  cv::Mat sample_resized;
  cv::Size input_geometry_(height, width);
  if (sample_clip.size() != input_geometry_){
    cv::resize(sample_clip, sample_resized, input_geometry_);
//	image_crop(smaple_float, sample_resized, height, width);
  }
  else
    sample_resized = sample_clip;

  //mobilenet
  //sample_resized = sample_resized.t();
  //cv::cvtColor(sample_resized, sample_resized, cv::COLOR_RGB2BGR);
  //printf("sample image data: %f ", *(sample_resized.ptr<float>(0)));

  cv::Mat sample_normalized;
  cv::subtract(sample_resized, mean, sample_normalized);

  sample_normalized = 0.017 * sample_normalized;

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == caffe_net.input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

typedef struct _dist
{
	int num;
	double interval;
	double max_output_value;
	double data[INTERVAL_NUM];
} Dist;

class LayerQuantData
{
public:
	LayerQuantData() {
		distribution.num = INTERVAL_NUM;
		distribution.max_output_value = 0;
		distribution.interval = 0;
		for (int j=0;j<INTERVAL_NUM;j++) distribution.data[j] = 0.0f;
	}

	Dist distribution;
	std::string layer_name;
	int layer_n;
	float activation_scale_factor;
	bool winograd;
	std::vector<float> weight_scale_factor;
};

double estimateKLDivergence(Dist &P, Dist &Q, double sum_P, double sum_Q)
{
	if (P.num != Q.num)
	{
		return 0.0f;
	}

	double divergence = 0;
	for (int i=0;i<P.num;i++)
	{
		P.data[i] /= sum_P;
		Q.data[i] /= sum_Q;
		if (P.data[i]>0 && Q.data[i]>0)
			divergence += P.data[i] * log(P.data[i] / Q.data[i]);
	}
	return divergence;
}


void extractMaxLayerOutput(Net<float> &caffe_net, std::string img_path, cv::Mat& mean,  std::vector<LayerQuantData> &quant_data)
{
	int img_num = 0;
	
	DIR* dp;
	struct dirent *dent;
	struct stat buf;
	if ((dp = opendir(img_path.c_str())) == NULL) {
		perror("cannot open val image directory");
		return;
	}
	if (img_path[img_path.length()-1] == '/') img_path.erase(img_path.length()-1);

	while((dent = readdir(dp))) {
		std::string file_name = img_path + "/" + dent->d_name;
		if ((stat(file_name.c_str(), &buf)) < 0) {
			perror("stat");
			std::cout<<"cannot open file: "<< file_name <<"\n";
			continue;
		}
		if (buf.st_mode & S_IFDIR) continue; //Directory
		if (file_name.find("jpg") == std::string::npos && file_name.find("JPEG") == std::string::npos)
			continue;
		img_num++;
	}

	FILE* f[80];
	for(int i=0;i<quant_data.size();i++) {
		char filename[80];
		sprintf(filename, "./out/dist_%d", i+1);
		f[i] = fopen(filename, "wb");
	}


	int iter = 0;
	rewinddir(dp);
	while((dent = readdir(dp))) {
		std::string file_name = img_path + "/" + dent->d_name;
		if ((stat(file_name.c_str(), &buf)) < 0) {
			perror("stat");
			std::cout<<"cannot open file: "<< file_name <<"\n";
			continue;
		}
		if (buf.st_mode & S_IFDIR) continue; //Directory
		if (file_name.find("jpg") == std::string::npos && file_name.find("JPEG") == std::string::npos)
			continue;

		cv::Mat img = cv::imread(file_name, -1);
			//invalid image
		if (img.rows <=0 || img.cols <=0)
		   	continue;

		Blob<float>* input_layer = caffe_net.input_blobs()[0];
		std::vector<cv::Mat> input_channels;
		WrapInputLayer(&input_channels, input_layer);
		Preprocess(caffe_net, img, mean, &input_channels, input_layer->channels(), input_layer->height(), input_layer->width());
		
		caffe_net.Forward();

		// Obtain max output
		std::vector<std::vector<Blob<float>*> > bottom_blobs = caffe_net.bottom_vecs();
		for (int i=0;i<quant_data.size();i++){
			Blob<float>* blob = bottom_blobs[quant_data[i].layer_n][0];
			int blob_size = blob->height() * blob->width() * blob->channels();
			const float *data_array = blob->cpu_data();
			for (int j=0;j<blob_size;j++, data_array++){
				float data = *data_array;
				//if (data<0) continue; //### have to do?
				//fprintf(f[i], "%f ", data);
				fwrite(&data, sizeof(float), 1, f[i]);
			}
		}

		iter++;
		printf("iter %d\n", iter);
	}

	for (int i=0;i<quant_data.size();i++){
		if(f[i]!=NULL)
			fclose(f[i]);
	}
}

int main(int argc, char** argv){
	::google::InitGoogleLogging(argv[0]);

	gflags::SetUsageMessage("Compute Scale Factors"
        "Usage:\n"
        "    compute_qunatization_factor PROTO_FILE INPUT_CAFFEMODEL MEAN_IMAGE CALIBRATION_IMAGE_DIRECTORY OUTPUT_DEPLOY OUTPUT_CAFFEMODEL\n");

	gflags::ParseCommandLineFlags(&argc, &argv, true);


	// Check args
	if (argc != 7) {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_quantization_factor");
		return 1;
	}

	// Caffe setting
	vector<int> gpus;
	get_gpus(&gpus);
	if (gpus.size() == 0) {
		Caffe::set_mode(Caffe::CPU);
	} else {
    	ostringstream s;
    	for (int i = 0; i < gpus.size(); ++i) {
    	  s << (i ? ", " : "") << gpus[i];
    	}
#ifndef CPU_ONLY
    	cudaDeviceProp device_prop;
    	for (int i = 0; i < gpus.size(); ++i) {
    	  cudaGetDeviceProperties(&device_prop, gpus[i]);
    	}
#endif
    	Caffe::SetDevice(gpus[0]);
    	Caffe::set_mode(Caffe::GPU);
	}


	// Read Net Model from Proto Files: ref. https://gist.github.com/onauparc/dd80907401b26b602885
	Net<float> caffe_net(argv[1], caffe::TEST);
	caffe_net.CopyTrainedLayersFrom(argv[2]);

	// Set mean image mat
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(argv[3], &blob_proto);

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
		for (int j=0; j<mean_blob.height()*mean_blob.width(); j++) { //Mobilenet
			if(i==0) *(data + j) = 103.94;
			else if(i==1) *(data + j) = 116.78;
			else if(i==2) *(data + j) = 123.68;
			else {
				exit(0);
			}
		}

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

	// List qunatization target layer
	std::vector<LayerQuantData> quant_data;

	const vector<shared_ptr<Layer<float> > > net_layers = caffe_net.layers();
	for (int i=0;i<net_layers.size();i++)
	{
		//Conv Layer decect
		if(strcmp(net_layers[i].get()->type(), "Convolution")==0){
			//caffe_net.
			//int output_num = caffe_net.top_vecs()[i][0]->shape()[1];
			LayerQuantData data;
			data.layer_name.assign(net_layers[i].get()->layer_param().name());
			data.layer_n = i;
			quant_data.push_back(data);
		}
	}

	// Collection Layer output
	extractMaxLayerOutput(caffe_net, argv[4], mean, quant_data);



	return 0;
}
#else
int main(int argc, char** argv(){
	LOG(FATAL) <<"This tool requires OpenCV; compile with USE_OPENCV.";
}
#endif // USE_OPENCV


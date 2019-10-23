#include <stdio.h>
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

  cv::Mat sample_resized;
  cv::Size input_geometry_(height, width);
  if (sample_float.size() != input_geometry_)
    cv::resize(sample_float, sample_resized, input_geometry_);
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
	std::vector<float> weight_scale_factor;
};

double estimateKLDivergence(Dist &P, Dist &Q, double sum_P, double sum_Q)
{
	if (P.num != Q.num)
	{
		std::cout<<"Can't compute KL Divergence: Not match Dimention\n";
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
	const fs::path path(img_path);

	std::cout<<"read calib image..\n";
	int img_num = 0;
	BOOST_FOREACH(const fs::path& p, std::make_pair(fs::directory_iterator(path), fs::directory_iterator()))
	{
		if (fs::is_directory(p)) continue;
		std::string file_name = p.string();
		if (file_name.find("jpg") == std::string::npos && file_name.find("JPEG") == std::string::npos)
			continue;
		img_num++;
	}

	int iter = 0;
	std::cout<<"extractMaxLayerOutput: ["<<iter<<"/"<<img_num<<"]\n";
	BOOST_FOREACH(const fs::path& p, std::make_pair(fs::directory_iterator(path), fs::directory_iterator()))
	{
		if (fs::is_directory(p)) continue;
		std::string file_name = p.string();
			// only jpg or JPEG file
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
			double max = quant_data[i].distribution.max_output_value;
			const float *data_array = blob->cpu_data();
			for (int j=0;j<blob_size;j++, data_array++){
				float data = *data_array;
				if (data<0) continue; //### have to do?
				if (max < data)
					max = data;
			}
			quant_data[i].distribution.max_output_value = max; 
		}

		iter++;
		std::cout<<"extractMaxLayerOutput: ["<<iter<<"/"<<img_num<<"]\n";
	}
}

void obtainDistribution(Net<float> &caffe_net, std::string img_path, cv::Mat& mean,  std::vector<LayerQuantData> &quant_data)
{
	double epsilon = 1.11e-16;
	for (int i=0;i<quant_data.size();i++)
	{
		quant_data[i].distribution.interval = (double)quant_data[i].distribution.max_output_value / (double)(INTERVAL_NUM-1) + epsilon;
		for(int j=0;j<INTERVAL_NUM;j++)
			quant_data[i].distribution.data[j] = 0;
	}

	const fs::path path(img_path);
	
	int img_num = 0;
	BOOST_FOREACH(const fs::path& p, std::make_pair(fs::directory_iterator(path), fs::directory_iterator()))
	{
		if (fs::is_directory(p)) continue;
		std::string file_name = p.string();
		if (file_name.find("jpg") == std::string::npos && file_name.find("JPEG") == std::string::npos)
			continue;
		img_num++;
	}

	int iter = 0;
	std::cout<<"ObtainDistribution: ["<<iter<<"/"<<img_num<<"]\n";
	BOOST_FOREACH(const fs::path& p, std::make_pair(fs::directory_iterator(path), fs::directory_iterator()))
	{
		if (fs::is_directory(p)) continue;
		std::string file_name = p.string();
			// only jpg or JPEG file
		if (file_name.find("jpg") == std::string::npos && file_name.find("JPEG") == std::string::npos)
			continue;

		cv::Mat img = cv::imread(file_name, -1);
		Blob<float>* input_layer = caffe_net.input_blobs()[0];
		std::vector<cv::Mat> input_channels;
		WrapInputLayer(&input_channels, input_layer);
		Preprocess(caffe_net, img, mean, &input_channels, input_layer->channels(), input_layer->height(), input_layer->width());
		caffe_net.Forward();

		/*
		 * Inference Testing
		Blob<float>* output_layer = caffe_net.output_blobs()[0];
		const float* begin = output_layer->cpu_data();
		const float* end = begin + output_layer->channels();
		std::vector<float> out(begin, end);
		int max = 0;
		for(int i=0;i<out.size();i++){
			if(out[i] > out[max]) max = i;
		}
		std::cout<<max<<"\n";*/

		// Obtain output distribution
		std::vector<std::vector<Blob<float>*> > bottom_blobs = caffe_net.bottom_vecs();
		for (int i=0;i<quant_data.size();i++){
			int sum_count=0;
			Blob<float>* blob = bottom_blobs[quant_data[i].layer_n][0];
			int blob_size = blob->height() * blob->width() * blob->channels();
			const float *data_array = blob->cpu_data();
			for (int j=0;j<blob_size;j++, data_array++){
				float data = *data_array;
				if (data<0) continue; //### have to do?
				//std::cout<<data<<" "<<((double)quant_data[i].distribution.max_output_value / (double)(INTERVAL_NUM-1))<<" "<<(int)((double)data * (double)(INTERVAL_NUM-1) / (double)quant_data[i].distribution.max_output_value)<<"\n";
				quant_data[i].distribution.data[(int)((double)data * (double)(INTERVAL_NUM-1) / (double)quant_data[i].distribution.max_output_value)]++;
				sum_count++;
			}
		}

		iter++;
		std::cout<<"ObtainDistribution: ["<<iter<<"/"<<img_num<<"]\n";
	}
}
/*
void obtainDistribution(std::vector<LayerQuantData> &quant_data)
{
	for(int i=0;i<quant_data.size();i++)
	{
		double interval = quant_data[i].distribution.interval = (double)quant_data[i].distribution.max_output.value / (double)(INTERVAL_NUM-1);
		quant_data[i].distribution.num = INTERVAL_NUM;
		for(int j=0;j<INTERVAL_NUM;j++) quant_data[i].distribution.data[j] = 0.0f;
		for(int j=0;j<quant_data[i].output_data.size();j++)
		{
			//if(i>9100) std::cout<<quant_data[i].output_data[j]<<" "<<interval<<":"<<(int)(quant_data[i].output_data[j] / interval)<<"\n";
			quant_data[i].distribution.data[(int)(quant_data[i].output_data[j] / interval)]++;
		}
	}
}*/

void computeActivationQuantize(std::vector<LayerQuantData> &quant_data)
{
	for (int layer=0;layer<quant_data.size();layer++)
	{
		quant_data[layer].distribution.data[0] = 0; //ReLU Zero Effect
//		std::cout<<layer<<"/"<<quant_data.size()<<" layer active quantize\n";
		int index = 0;
		double min_divergence = 0;
		double last = 0.0f;
		for (int j=128;j<INTERVAL_NUM;j++){
			last+=quant_data[layer].distribution.data[j];
		}
		for (int i = 128; i<INTERVAL_NUM; i++)
		{
			Dist P, candidate_Q, Q;
			double sum_P = 0.0f;
			double sum_Q = 0.0f;

			for (int j=0;j<i;j++)
			{
				P.data[j] = quant_data[layer].distribution.data[j];
				sum_P += quant_data[layer].distribution.data[j];
			}
//			P.data[i-1] += last;
//			sum_P += last;
//			last -= quant_data[layer].distribution.data[i];
			for (int j=i;j<INTERVAL_NUM;j++)
			{
				P.data[i-1] += quant_data[layer].distribution.data[j];
				sum_P += quant_data[layer].distribution.data[j];
			}

			//int nzero_count[128];
			//for(int j=0;j<128;j++){ candidate_Q.data[j] = 0.0f; nzero_count[j] = 0; }
			for (int j=0;j<128;j++){ candidate_Q.data[j] = 0.0f; }
			/*for(int j=0;j<i;j++)
			{
				int l = (int)((double)j/(double)i*128.0);
				candidate_Q.data[l] += quant_data[layer].distribution.data[j];
				if(quant_data[layer].distribution.data[j]!=0) nzero_count[l]++;
			}
			for(int j=0;j<i;j++)
			{
				Q.data[j] = 0.0f;
				int l = (int)((double)j/(double)i*128.0);
				if(nzero_count[l] != 0 && P.data[j] != 0)
				{
					Q.data[j] = candidate_Q.data[l] / (double)nzero_count[l];
				}
				sum_Q += Q.data[j];
			}*/
			int num_merged_bins = (double)i / (double)128.0;
			for (int j=0;j<128;j++){
				int start = j * num_merged_bins;
				int stop = start + num_merged_bins;
				for (int k=start;k<stop;k++){
					candidate_Q.data[j] += quant_data[layer].distribution.data[k];
				}
			}
			for (int k=128*num_merged_bins;k<i;k++) candidate_Q.data[127] += quant_data[layer].distribution.data[k];
			for (int j=0;j<128;j++){
				int start = j * num_merged_bins;
				int stop = (j==127)?(i):(start + num_merged_bins);
				int norm = 0;
				for (int k=start;k<stop;k++){
					if(P.data[k]!=0) norm++;
				}
				for (int k=start;k<stop;k++){
					Q.data[k] = 0.0f;
					if(norm!=0 && P.data[k]!=0)
						Q.data[k] = (double)candidate_Q.data[j] / (double)norm;
				}
			}
			sum_P = 0;
			sum_Q = 0;
			for(int j=0;j<i;j++){
				if(P.data[j]==0) P.data[j]=0.0001;
				if(Q.data[j]==0) Q.data[j]=0.0001;
				sum_P += P.data[j];
				sum_Q += Q.data[j];
			}
/*
			// scale - threshold
			int nzero_count[128];
			Dist dist_Q, real_Q;
			for(int j=0;j<128;j++){ dist_Q.data[j] = 0.0f; nzero_count[j] = 0; }
			for(int j=0;j<i;j++){
				int l = (int)((double)j/(double)i*128.0);
				dist_Q.data[l] += quant_data[layer].distribution.data[j];
				if(quant_data[layer].distribution.data[j]!=0) nzero_count[l]++;
			}
			for(int j=i;j<INTERVAL_NUM;j++){
				dist_Q.data[127] += quant_data[layer].distribution.data[j];
				if(quant_data[layer].distribution.data[j]!=0) nzero_count[127]++;
			}
			for(int j=0;j<INTERVAL_NUM;j++){ real_Q.data[j] = 0.0f; }
			for(int j=0;j<i;j++){
				int l = (int)((double)j/(double)i*128.0);
				if(nzero_count[l] != 0 && quant_data[layer].distribution.data[j]!=0)
					real_Q.data[j] = dist_Q.data[l] / (double)nzero_count[l];
				sum_Q += real_Q.data[j];
			}*/

			/*if(i==258){
			for(int j=0;j<i;j++){
				std::cout<<P.data[j]<<" ";
			}
			std::cout<<"\n";
			for(int j=0;j<i;j++){
				std::cout<<Q.data[j]<<" ";
			}
			std::cout<<"\n";
			scanf("%d", &P.num);
			}*/
			P.num = i;
			Q.num = i;
			//real_Q.num = INTERVAL_NUM;
			
			double divergence = estimateKLDivergence(P, Q, sum_P, sum_Q);
			//double divergence = estimateKLDivergence(real_Q, quant_data[layer].distribution, sum_Q,  sum_P);

			if (divergence < min_divergence || index==0)
			{
				min_divergence = divergence;
				index = i;
			}
		}
		
		double threshold = ((double)index + (double)0.5) * quant_data[layer].distribution.interval;
		quant_data[layer].activation_scale_factor = 127.0 / (float)threshold;

		std::cout<<"Activation Qunatizing: ["<<layer<<"/"<<quant_data.size()<<"]\n";
		std::cout<<min_divergence<<" "<<index<<"\n";
	}
}


/** for cmath fault in gcc.5 */
inline float abs(float a){
	if(a<0) return -a;
	else return a;
}

void computeWeightQuantize(Net<float> &caffe_net, std::vector<LayerQuantData> &quant_data)
{
	//int op[500] = {0};
	//op[13]=op[26]=op[39]=op[55]=op[68]=op[81]=op[94]=op[110]=op[123]=op[136]=op[149]=op[162]=op[175]=op[191]=op[204]=op[217] = 1;
	std::cout<<"Weight Qunatizing: ["<<0<<"/"<<quant_data.size()<<"]\n";
	for (int i=0;i<quant_data.size();i++)
	{
		Blob<float>* weight = caffe_net.layers()[quant_data[i].layer_n].get()->blobs()[0].get();
		int height = weight->height();
		int width = weight->width();
		int channels = weight->channels();
		for (int j=0;j<weight->num();j++){
			float max = 0;
			const float *data_array = weight->cpu_data() + (j * height*width*channels);
			for (int r=0;r<height*width*channels;r++){
				float data = *data_array;
				if (max < abs(data)){
					max = abs(data);
				}
				data_array++;
			}
			quant_data[i].weight_scale_factor.push_back(0);
			if (max < 0.0001) {
				quant_data[i].weight_scale_factor[j] = 0.0f;
			}
			else {
				//const ConvolutionParameter param = caffe_net.layers()[quant_data[i].layer_n].get()->layer_param().convolution_param();
				//if(param.kernel_h() == 3 && param.kernel_w() == 3 && param.stride_h() == 1 && param.stride_w() == 1 && param.num_output() != param.group()){
				//   	quant_data[i].weight_scale_factor[j] = 31.0f / max;
				//	std::cout<<"winograd\n";
				//}
				//else 
					quant_data[i].weight_scale_factor[j] = 127.0f / max;
			}
		}
		CHECK_EQ(quant_data[i].weight_scale_factor.size(), weight->num()) << "Weight scale factor num error: versus weight output num: "<<quant_data[i].weight_scale_factor.size()<<" vs "<<weight->num();
		std::cout<<"Weight Qunatizing: ["<<i<<"/"<<quant_data.size()<<"]\n";
	}
}

void SaveQuantizedDeploy(std::string src_deploy_path, std::vector<LayerQuantData> &quant_data, std::string dest_deploy_path)
{
	NetParameter net_param;
	ReadProtoFromTextFile(src_deploy_path, &net_param);

	for (int i = 0; i < net_param.layer_size(); i++) {
		for (int j = 0; j < quant_data.size(); j++) {
			if ( !(net_param.layer(i).name().compare(quant_data[j].layer_name) == 0) ) continue;
			LayerParameter* layer_param = net_param.mutable_layer(i);
			layer_param->clear_int8_inference();
			//layer_param->clear_activation_scale_factor();
			//layer_param->clear_weight_scale_factor();

			layer_param->set_int8_inference(true);
			//layer_param->set_activation_scale_factor(quant_data[j].activation_scale_factor);
			//for (int k = 0; k < quant_data[j].weight_scale_factor.size(); k++) {
			//	layer_param->add_weight_scale_factor(quant_data[j].weight_scale_factor[k]);
			//}
		}
	}
	WriteProtoToTextFile(net_param, dest_deploy_path);
}

void SaveQuantizedModel(std::string src_deploy_path, std::string src_model_path, std::vector<LayerQuantData> &quant_data, std::string dest_model_path)
{
	NetParameter net_param;
	ReadProtoFromTextFile(src_deploy_path, &net_param);
	ReadProtoFromBinaryFile(src_model_path, &net_param);

	for (int i = 0; i < net_param.layer_size(); i++) {
		for (int j = 0; j < quant_data.size(); j++) {
			if ( !(net_param.layer(i).name().compare(quant_data[j].layer_name) == 0) ) continue;
			LayerParameter* layer_param = net_param.mutable_layer(i);
			layer_param->clear_activation_scale_factor();
			layer_param->clear_weight_scale_factor();
			
			layer_param->set_activation_scale_factor(quant_data[j].activation_scale_factor);
			for (int k = 0; k < quant_data[j].weight_scale_factor.size(); k++) {
				layer_param->add_weight_scale_factor(quant_data[j].weight_scale_factor[k]);
			}

			// Conv Weight Quantize
			BlobProto* blob = layer_param->mutable_blobs(0);

			//std::cout<<"int8 weights: "<<net_param.layer(i).name()<<": "<<blob->data_size()<<"\n";

			blob->clear_int8_data();

			std::string int8_data;
			int8_data.reserve(blob->data_size());
			int channels;
			int height;
			int width;
			int length;
			if (blob->has_shape()) {
				channels = blob->shape().dim(1);
				height = blob->shape().dim(2);
				width = blob->shape().dim(3);
				length = channels * height * width;
			}
			else {
				channels = blob->channels();
				height = blob->height();
				width = blob->width();
				length = channels * height * width;
			}
			//std::cout<<blob->shape().dim(0)<<" "<<blob->shape().dim(1)<<" "<<blob->shape().dim(2)<<" "<<blob->shape().dim(3)<<"\n";
			//std::cout<<blob->data_size()<<" "<<length<<" "<<quant_data[j].weight_scale_factor.size()<<"\n";
			for (int k = 0; k < blob->data_size(); k++) {
				float fp32_weight = blob->data(k);
				float quantized_weight = (fp32_weight * quant_data[j].weight_scale_factor[k/length]);
				if ( quantized_weight >= 0.f) quantized_weight += 0.5;
				else quantized_weight -= 0.5;

				if ( quantized_weight > 127 ) {
					quantized_weight = 127;
				}
				else if ( quantized_weight < -128 ) {
					quantized_weight = -128;
				}
				char int8_weight = (char)quantized_weight;
				int8_data.push_back(int8_weight);

				//std::cout<<(int)int8_weight<<" ";
			}
			//std::cout<<"\n";

			blob->set_int8_data(int8_data);

			blob->clear_data();
			blob->clear_diff();
			blob->clear_double_data();
			blob->clear_double_diff();
		}
	}

	WriteProtoToBinaryFile(net_param, dest_model_path);
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
		LOG(INFO) << "Use CPU.";
		std::cout<<"Use CPU.\n";
		Caffe::set_mode(Caffe::CPU);
	} else {
    	ostringstream s;
    	for (int i = 0; i < gpus.size(); ++i) {
    	  s << (i ? ", " : "") << gpus[i];
    	}
    	LOG(INFO) << "Using GPUs " << s.str();
		std::cout<<"Use GPUs.\n";
#ifndef CPU_ONLY
    	cudaDeviceProp device_prop;
    	for (int i = 0; i < gpus.size(); ++i) {
    	  cudaGetDeviceProperties(&device_prop, gpus[i]);
    	  LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
    	}
#endif
    	Caffe::SetDevice(gpus[0]);
    	Caffe::set_mode(Caffe::GPU);
	}


	// Read Net Model from Proto Files: ref. https://gist.github.com/onauparc/dd80907401b26b602885
	std::cout<<"read Net Model..\n";
	Net<float> caffe_net(argv[1], caffe::TEST);
	caffe_net.CopyTrainedLayersFrom(argv[2]);

	// Set mean image mat
	std::cout<<"set mean image mat..\n";
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

	// Obtain Distribution
	obtainDistribution(caffe_net, argv[4], mean, quant_data);

	// Activation Quantize;
	computeActivationQuantize(quant_data);

	// Weight Quantize
	computeWeightQuantize(caffe_net, quant_data);

	std::ofstream out("scale_i.table");
	for (int i=0;i<quant_data.size();i++)
	{
		out<<caffe_net.layer_names()[quant_data[i].layer_n]<<"_param_0 ";
		for(int j=0;j<quant_data[i].weight_scale_factor.size();j++)
			out<<quant_data[i].weight_scale_factor[j]<<" ";
		out<<"\n";
	}
	for (int i=0;i<quant_data.size();i++)
	{
		out<<caffe_net.layer_names()[quant_data[i].layer_n]<<" "<<quant_data[i].activation_scale_factor<<"\n";
	}

	// Save Quantization Results To deploy proto & caffemodel
	SaveQuantizedDeploy(argv[1], quant_data, argv[5]);
	SaveQuantizedModel(argv[1], argv[2], quant_data, argv[6]);

	return 0;
}
#else
int main(int argc, char** argv(){
	LOG(FATAL) <<"This tool requires OpenCV; compile with USE_OPENCV.";
}
#endif // USE_OPENCV


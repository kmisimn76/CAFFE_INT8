#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <cstring> 
#include <fstream> 
#include <iostream> 
#include <string> 
#include <vector> 

#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

using caffe::Caffe;

using namespace caffe;


int main(int argc, char** argv) {
	int tmp;
	if(argc!=3){
		std::cout<<"parameter error\n";
	}
	std::string deploy(argv[1]);
	std::string model(argv[2]);

	Caffe::set_mode(Caffe::CPU);
	Net<float> caffe_net(deploy, caffe::TEST);
	caffe_net.CopyTrainedLayersFrom(model);

	
	FILE* f;

	f = fopen("weight", "wb");

	//Extract Weight
	//
	//Format :
	//
	//4bytes(int)	: # of layer
	//
	//30bytes(char)	: layer name(caffe)
	//30bytes(char)	: layer type(caffe)
	//4bytes(int)	: 0 - FP32 INFER, 1 - INT8 INFER
	//4bytes(int)	: # of blobs(weights)
	//{
	//	16bytes(int*4)	: shape: nchw
	//	n-bytes(float|int * weight size) : weight
	//}* # of blobs repeat
	//if(int8 infer){
	//	4bytes(float): activation scale factor
	//	n-bytes(float * # of filter) : weight scale factor
	//}
	//
#define INT8_INFERENCE (&(tmp=1))
#define FP32_INFERENCE (&(tmp=0))
	const vector<shared_ptr<Layer<float> > > net_layers = caffe_net.layers();
	fwrite(&(tmp=net_layers.size()), sizeof(int), 1, f);
	printf("%d\n", net_layers.size());
	for (int i=0;i<net_layers.size();i++)
	{
		if(net_layers[i]->layer_param().int8_inference() == true) {
			//fwrite((&(tmp=caffe_net.layers()[i].get()->int8_blobs().size())), sizeof(int), 1, f);
			//printf("%ld\n", caffe_net.layers()[i].get()->int8_blobs().size());
			//for(int j=0;j<caffe_net.layers()[i].get()->int8_blobs().size(); j++)
			char layer_name[81];
			char layer_type[81];
			strcpy(layer_name, net_layers[i]->layer_param().name().c_str());
			strcpy(layer_type, net_layers[i]->layer_param().type().c_str());
			fwrite(layer_name, sizeof(char), 80, f);
			printf("%s\n", layer_name);
			fwrite(layer_type, sizeof(char), 80, f);
			printf("%s\n", layer_type);

			fwrite(INT8_INFERENCE, sizeof(int), 1, f);
			printf("%d ", 1);
			int num_blob = ( caffe_net.layers()[i]->layer_param().convolution_param().bias_term())?(2):(1);
			fwrite((&(num_blob)), sizeof(int), 1, f);
			printf("%ld\n", num_blob);
			for(int j=0;j<1; j++)
			{
				Blob<char>* weight = caffe_net.layers()[i].get()->int8_blobs()[j].get();
				int shape[4] = {weight->num(), weight->channels(), weight->height(), weight->width()};
				const signed char* data_array = (const signed char*)weight->cpu_data();

				fwrite(shape, sizeof(int), 4, f);
				//printf("%d %d %d %d ", shape[0], shape[1], shape[2], shape[3]);
				fwrite(data_array, sizeof(signed char), shape[0]*shape[1]*shape[2]*shape[3], f);
				for(int jj=0;jj<shape[0]*shape[1]*shape[2]*shape[3];jj++) {
					//printf("%c ", data_array[jj]);
				}
			}
			printf("\n");
			for(int j=0;j<1 && num_blob==2; j++)
			{
				Blob<int>* weight = caffe_net.layers()[i].get()->int_blobs()[j].get();
				int shape[4] = {weight->num(), weight->channels(), weight->height(), weight->width()};
				const int* data_array = (const int*)weight->cpu_data();

				fwrite(shape, sizeof(int), 4, f);
				//printf("%d %d %d %d ", shape[0], shape[1], shape[2], shape[3]);
				fwrite(data_array, sizeof(int), shape[0]*shape[1]*shape[2]*shape[3], f);
				for(int jj=0;jj<shape[0]*shape[1]*shape[2]*shape[3];jj++) {
					//printf("%d ", data_array[jj]);
				}
			}
			printf("\n");
			float act_scale_factor = caffe_net.layers()[i].get()->activation_scale_factor();
			float* weight_scale_factor = &(caffe_net.layers()[i].get()->weight_scale_factor())[0];
			fwrite(&act_scale_factor, sizeof(float), 1, f);
			printf("%f ", act_scale_factor);
			int nu_w = (caffe_net.layers()[i].get()->int8_blobs()[0].get())->num();
			fwrite(&nu_w, sizeof(int), 1, f);
			printf("%d ", nu_w);
			fwrite(weight_scale_factor, sizeof(float), nu_w, f); //weight_scale_factor.size() == weight->num()
			for(int l=0;l<nu_w;l++) {
				printf("%f ", weight_scale_factor[l]);
			}
			printf("\n");
		}
		else {
			char layer_name[81];
			char layer_type[81];
			strcpy(layer_name, net_layers[i]->layer_param().name().c_str());
			strcpy(layer_type, net_layers[i]->layer_param().type().c_str());
			fwrite(layer_name, sizeof(char), 80, f);
			printf("%s\n", layer_name);
			fwrite(layer_type, sizeof(char), 80, f);
			printf("%s\n", layer_type);

			fwrite(FP32_INFERENCE, sizeof(int), 1, f);
			printf("%d ", 0);
			int num_blobs = 0;
			if(strcmp(layer_type, "Convolution")==0) num_blobs = (caffe_net.layers()[i]->layer_param().convolution_param().bias_term())?(2):(1);
			fwrite((&(tmp=num_blobs)), sizeof(int), 1, f);
//			printf("%ld\n", num_blobs);
			for(int j=0;j<num_blobs; j++)
			{
				Blob<float>* weight = caffe_net.layers()[i].get()->blobs()[j].get();
				int shape[4] = {weight->num(), weight->channels(), weight->height(), weight->width()};
				const float* data_array = weight->cpu_data();

				fwrite(shape, sizeof(int), 4, f);
//				printf("%d %d %d %d ", shape[0], shape[1], shape[2], shape[3]);
				fwrite(data_array, sizeof(float), shape[0]*shape[1]*shape[2]*shape[3], f);
				for(int jj=0;jj<shape[0]*shape[1]*shape[2]*shape[3];jj++) {
//					printf("%f ", data_array[jj]);
				}
			}
//			printf("\n");
		}
	}
	fclose(f);
	
	return 0;
}

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Automatically created from a TensorFlow Lite flatbuffer using the command:
// xxd -i model.tflite > model.cc

// This is a standard TensorFlow Lite model file that has been converted into a
// C data array, so it can be easily compiled into a binary for devices that
// don't have a file system.

// See train/README.md for a full description of the creation process.

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MODEL_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MODEL_H_

#define USE_LCD 0

#define USE_FEATURES 0
#define USE_ENCODER 0

#define USE_NN 1

#if USE_FEATURES
extern const unsigned char feature_nn_tflite[];
extern unsigned int feature_nn_tflite_len;
#elif USE_ENCODER
extern const unsigned char encoder_tflite[];
extern unsigned int encoder_tflite_len;
#elif USE_NN
extern const unsigned char cnn_quant_int_tflite[];
extern unsigned int cnn_quant_int_tflite_len;
#else
extern const unsigned char clf_tflite[];
extern unsigned int clf_tflite_len;
#endif

#define NUM_CLASSES 12
#define NUM_FEATURES 14
//#define NUM_DATA 3794
#define NUM_DATA 1000

#define WINDOW_SIZE 256
#define NUM_RAW_DATA 1
//#define NUM_LABELS 25
#define NUM_LABELS 12

extern const float data[NUM_DATA][NUM_FEATURES];

extern const float raw_data[NUM_RAW_DATA][WINDOW_SIZE];


#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MODEL_H_

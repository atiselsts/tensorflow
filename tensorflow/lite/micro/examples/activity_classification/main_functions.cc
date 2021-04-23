#include "tensorflow/lite/micro/examples/activity_classification/main_functions.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/examples/activity_classification/constants.h"
#include "tensorflow/lite/micro/examples/activity_classification/model.h"
#include "tensorflow/lite/micro/examples/activity_classification/output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <math.h>

#if USE_LCD
#include "mbed.h"
#include "LCD_DISCO_F746NG.h"

extern LCD_DISCO_F746NG lcd;
#endif

extern tflite::ErrorReporter* error_reporter;

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

//constexpr int kTensorArenaSize = 2000;
constexpr int kTensorArenaSize = 300000;

uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {

#if USE_LCD
    char buf[100];

    lcd.DisplayStringAt(0, LINE(1), (uint8_t *)"setup: get model", CENTER_MODE);
#endif

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
#if USE_FEATURES
  model = tflite::GetModel(feature_nn_tflite);
#elif USE_ENCODER
  model = tflite::GetModel(encoder_tflite);
#elif USE_NN
  model = tflite::GetModel(cnn_quant_int_tflite);
#else
  model = tflite::GetModel(clf_tflite);
#endif

  // sprintf(buf, "model: check version %p", model);
  // lcd.DisplayStringAt(0, LINE(2), (uint8_t *)buf, CENTER_MODE);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // sprintf(buf, "allocate mem");
  // lcd.DisplayStringAt(0, LINE(3), (uint8_t *)buf, CENTER_MODE);

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
      return;
  }

  // sprintf(buf, "allocate status: %d", allocate_status);
  // lcd.DisplayStringAt(0, LINE(4), (uint8_t *)buf, CENTER_MODE);

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Calculate an x value to feed into the model. We compare the current
  // inference_count to the number of inferences per cycle to determine
  // our position within the range of possible x values the model was
  // trained on, and use this to calculate a value.
  float position = static_cast<float>(inference_count) /
                   static_cast<float>(kInferencesPerCycle);
  float x = position * kXrange;


  const float *f = data[inference_count * 5];
  int i;
  for (i = 0; i < NUM_FEATURES; ++i) {
    input->data.f[i] = f[i];
  }

  // // Quantize the input from floating-point to integer
  // int8_t x_quantized = x / input->params.scale + input->params.zero_point;
  // // Place the quantized input in the model's input tensor
  // input->data.int8[0] = x_quantized;

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x: %f\n",
                         static_cast<double>(x));
      return;
  }

  // // Obtain the quantized output from model's output tensor
  // int8_t y_quantized = output->data.int8[0];
  // // Dequantize the output from integer to floating-point
  // float y = (y_quantized - output->params.zero_point) * output->params.scale;

  int best_class = 0;
  TF_LITE_REPORT_ERROR(error_reporter, "output\n");
  for (i = 0; i < NUM_CLASSES; ++i) {
//      TF_LITE_REPORT_ERROR(error_reporter, "  out[%d]=%f\n", i,  output->data.f[i]);
      if (output->data.f[i] > output->data.f[best_class]) {
          best_class = i;
      }
  }

  float y = (float)best_class / NUM_CLASSES;
  // Output the results. A custom HandleOutput function can be implemented
  // for each supported hardware target.
  HandleOutput(error_reporter, x, y);

  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) inference_count = 0;
}


int nn_classify_single(const float features[])
{
  const float *f = features;
  int i;
  for (i = 0; i < NUM_FEATURES; ++i) {
    input->data.f[i] = f[i];
  }

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
      return -1;
  }

  int best_class = 0;
  for (i = 0; i < NUM_CLASSES; ++i) {
      if (output->data.f[i] > output->data.f[best_class]) {
          best_class = i;
      }
  }

  return best_class;
}

int nn_classify_single_from_data(const float data[])
{
  // pass input data
  // lcd.DisplayStringAt(0, LINE(1), (uint8_t *)"set input data", CENTER_MODE);

#if USE_LCD
  const int rows = input->dims->data[1];
  const int cols = input->dims->data[2];
  char buf[100];
  sprintf(buf, "dims = %u (%d %d %d %d)", input->dims->size,
          input->dims->data[0], input->dims->data[1], input->dims->data[2], input->dims->data[3]);
  lcd.DisplayStringAt(0, LINE(10), (uint8_t *)buf, CENTER_MODE);
#endif

  int i;

  // const float *f = data;
  // for (i = 0; i < WINDOW_SIZE; ++i) {
  //     input->data.f[i] = f[i];
  // }
  for (i = 0; i < WINDOW_SIZE * 3; ++i) {
      input->data.int8[i] = (int8_t)i;
  }

  // for (i = 0; i < WINDOW_SIZE; ++i) {
  //     input->data.int8[i] = (int8_t)(i * 3);
  //     input->data.int8[i + WINDOW_SIZE] = (int8_t)(i * 3 + 1);
  //     input->data.int8[i + WINDOW_SIZE * 2] = (int8_t)(i * 3 + 2);
  // }

  // lcd.DisplayStringAt(0, LINE(2), (uint8_t *)"input data set", CENTER_MODE);

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  // lcd.DisplayStringAt(0, LINE(3), (uint8_t *)"invoke done", CENTER_MODE);

  if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
      return -1;
  }


  // select the best class
  int best_class = 0;
  for (i = 0; i < NUM_LABELS; ++i) {
//      if (output->data.f[i] > output->data.f[best_class]) {
      if (output->data.int8[i] > output->data.int8[best_class]) {
          best_class = i;
      }
  }

//  sprintf(buf, "class=%d val=%d", best_class, output->data.int8[best_class]);
#if USE_LCD
  sprintf(buf, "%d %d %d %d %d %d",
          output->data.int8[0], output->data.int8[1], output->data.int8[2],
          output->data.int8[3], output->data.int8[4], output->data.int8[5]);
  lcd.DisplayStringAt(0, LINE(4), (uint8_t *)buf, CENTER_MODE);
  sprintf(buf, "%d %d %d %d %d %d",
          output->data.int8[6], output->data.int8[7], output->data.int8[8],
          output->data.int8[9], output->data.int8[10], output->data.int8[11]);
  lcd.DisplayStringAt(0, LINE(5), (uint8_t *)buf, CENTER_MODE);
#endif

  return best_class;
}

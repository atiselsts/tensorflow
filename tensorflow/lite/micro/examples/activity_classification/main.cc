
#include "tensorflow/lite/micro/examples/activity_classification/main_functions.h"
#include "tensorflow/lite/micro/examples/activity_classification/model.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

#include "mbed.h"
#include "LCD_DISCO_F746NG.h"

tflite::ErrorReporter* error_reporter = nullptr;

/*

To build and upload run:

make -f tensorflow/lite/micro/tools/make/Makefile TARGET=mbed TAGS="CMSIS disco_f746ng" generate_activity_classification_mbed_project

cd tensorflow/lite/micro/tools/make/gen/mbed_cortex-m4/prj/activity_classification/mbed

mbed compile -m DISCO_F746NG -t GCC_ARM && cp ./BUILD/DISCO_F746NG/GCC_ARM/mbed.bin /media/atis/DIS_F746NG/

*/


LCD_DISCO_F746NG lcd;

#define NUM_TESTS 1

int nn_classify(void)
{
    int i;
    int dummy = 0;
#if USE_FEATURES
    for (i = 0; i < NUM_DATA; ++i) {
        dummy += nn_classify_single(data[i]);        
    }
#else
    for (i = 0; i < NUM_RAW_DATA; ++i) {
        dummy += nn_classify_single_from_data(raw_data[i]);        
    }
#endif
//    TF_LITE_REPORT_ERROR(error_reporter, "classified from C++!\n");
//    lcd.DisplayStringAt(0, LINE(2), (uint8_t *)"classified from C++!", CENTER_MODE);
    return dummy;
}

void classify(void)
{
    uint64_t start = get_ms_count();
    int i;
    int dummy = 0;
    for (i = 0; i < NUM_TESTS; ++i) {
        dummy += nn_classify();
    }
    uint64_t delta = get_ms_count() - start;

    char buf[100];
    sprintf(buf, "%u tests: %llu ms (%d)", NUM_TESTS, delta, dummy);
    TF_LITE_REPORT_ERROR(error_reporter, buf);
    lcd.DisplayStringAt(0, LINE(4), (uint8_t *)buf, CENTER_MODE);
}

// This is the default main used on systems that have the standard C entry
// point. Other devices (for example FreeRTOS or ESP32) that have different
// requirements for entry code (like an app_main function) should specialize
// this main.cc file in a target-specific subfolder.
int main(int argc, char* argv[]) {

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  TF_LITE_REPORT_ERROR(error_reporter,
          "Hello World from activity classification example!");

  lcd.Clear(LCD_COLOR_WHITE);
  lcd.SetTextColor(LCD_COLOR_BLACK);
  lcd.DisplayStringAt(0, LINE(0), (uint8_t *)"Activity class. example", CENTER_MODE);

    setup();
  //  while (true) {
  //    loop();
  //  }

    lcd.DisplayStringAt(0, LINE(1), (uint8_t *)"Setup done", CENTER_MODE);

//    classify();
}

ACTIVITY_CLASSIFICATION_TEST_SRCS := \
tensorflow/lite/micro/examples/activity_classification/activity_classification_test.cc \
tensorflow/lite/micro/examples/activity_classification/model.cc

ACTIVITY_CLASSIFICATION_TEST_HDRS := \
tensorflow/lite/micro/examples/activity_classification/model.h

ACTIVITY_CLASSIFICATION_SRCS := \
tensorflow/lite/micro/examples/activity_classification/main.cc \
tensorflow/lite/micro/examples/activity_classification/main_functions.cc \
tensorflow/lite/micro/examples/activity_classification/model.cc \
tensorflow/lite/micro/examples/activity_classification/output_handler.cc \
tensorflow/lite/micro/examples/activity_classification/constants.cc

ACTIVITY_CLASSIFICATION_HDRS := \
tensorflow/lite/micro/examples/activity_classification/model.h \
tensorflow/lite/micro/examples/activity_classification/output_handler.h \
tensorflow/lite/micro/examples/activity_classification/constants.h \
tensorflow/lite/micro/examples/activity_classification/main_functions.h

#Find any platform - specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/activity_classification/*/Makefile.inc)

# Tests loading and running the sine model.
$(eval $(call microlite_test,activity_classification_test,\
$(ACTIVITY_CLASSIFICATION_TEST_SRCS),$(ACTIVITY_CLASSIFICATION_TEST_HDRS)))

# Builds a standalone binary.
$(eval $(call microlite_test,activity_classification,\
$(ACTIVITY_CLASSIFICATION_SRCS),$(ACTIVITY_CLASSIFICATION_HDRS)))

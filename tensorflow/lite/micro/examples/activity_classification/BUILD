# Description:
#   On-board activity classification using TensorFlow Lite for Microcontrollers.

load(
    "//tensorflow/lite/micro:build_def.bzl",
    "micro_copts",
)

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "model",
    srcs = [
        "model.cc",
    ],
    hdrs = [
        "model.h",
    ],
    copts = micro_copts(),
)

cc_library(
    name = "output_handler",
    srcs = [
        "output_handler.cc",
    ],
    hdrs = [
        "output_handler.h",
    ],
    copts = micro_copts(),
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_error_reporter",
    ],
)

cc_library(
    name = "constants",
    srcs = [
        "constants.cc",
    ],
    hdrs = [
        "constants.h",
    ],
    copts = micro_copts(),
)

cc_binary(
    name = "activity_classification",
    srcs = [
        "main.cc",
        "main_functions.cc",
        "main_functions.h",
    ],
    copts = [
        "-Werror",
        "-Wdouble-promotion",
        "-Wsign-compare",
    ],
    deps = [
        ":constants",
        ":model",
        ":output_handler",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

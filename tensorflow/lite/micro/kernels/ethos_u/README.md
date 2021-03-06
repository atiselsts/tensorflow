<!-- mdformat off(b/169948621#comment2) -->

# Info
Arm(R) Ethos(TM)-U is a new class of machine learning processors, called a
microNPU, specifically designed to accelerate ML inference in area-constrained
embedded and IoT devices. This readme briefly describes how to integrate Ethos-U
related hardware and software into TFLM.

To enable the Ethos-U software stack, add `CO_PROCESSOR=ethos_u` to the make
command line. See example below.

## Requirements:
- Armclang 6.14 or later
- GCC 10.2.1 or later

## Ethos-U custom operator
The TFLM runtime will dispatch workloads to Ethos-U when it encounters an
Ethos-U custom op in the tflite file. The Ethos-U custom op is added by a tool
called Vela and contains information the Ethos-U hardware need to execute
the workload. More info in the [Vela repo](https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ethos-u-vela).

```
     | tensor0
     |
     v
+------------+
| ethos-u    |
| custom op  |
+------------+
     +
     |
     | tensor1
     |
     v
+-----------+
| transpose |
|           |
+----|------+
     |
     | tensor2
     |
     v
```

Note that the `ethousu_init()` API of the Ethos-U driver need to be called at
startup, before calling the TFLM API. More info in the [Ethos-U driver repo](https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ethos-u-core-driver).

For even more info regarding Vela and Ethos-U, checkout [Ethos-U landing page](https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ethos-u/+/refs/heads/master).

# Example 1

Compile a binary with Ethos-U support using the following command:

```
make -f tensorflow/lite/micro/tools/make/Makefile network_tester_test CO_PROCESSOR=ethos_u \
TARGET=<ethos_u_enabled_target> NETWORK_MODEL=<ethos_u_enabled_tflite>
```

TODO: Replace `ethos_u_enabled_target` and `ethos_u_enabled_tflite` once the
Arm Corstone(TM)-300 example is up and running.

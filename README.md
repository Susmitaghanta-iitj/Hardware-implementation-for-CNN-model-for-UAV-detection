This thesis presents the complete hardware design and FPGA/ASIC implementation
of a Convolutional Neural Network (CNN) based perception module for Unmanned
Aerial Vehicle (UAV) applications using Register Transfer Level (RTL) methodology.
The objective of the work is to translate a floating-point UAV vision model into a
fully synthesizable, resource-efficient digital hardware architecture capable of realtime inference on edge platforms. The proposed system replaces software-based inference with a custom hardware pipeline consisting of Conv1D processing blocks, batchnormalization, activation functions, max-pooling, flattening, and multi-layer fully connected classifiers. All algorithmic layers are redesigned in fixed-point Q4.12 format
and optimized for FPGA constraints through parallel MAC units, pipelined datapath,
weight quantization, and BRAM-based memory organization. The complete CNN
architecture is implemented in Verilog RTL, mapped to FPGA fabric, and validated
through functional simulation and post-synthesis verification. The design supports continuous streaming of sensor data and achieves deterministic latency, making it suitable
for onboard UAV embedded systems where low power, fast response, and real-time
decision making are critical. The results demonstrate that the FPGA-based implementation offers significant improvements in throughput and energy efficiency compared
to CPU/GPU software implementations, highlighting the feasibility of deploying deeplearning-based UAV intelligence on low-power reconfigurable hardware.

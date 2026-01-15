# Hardware-implementation-for-CNN-model-for-UAV-detection
`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/21/2025 03:46:00 PM
// Design Name: 
// Module Name: cnn_model_layer1
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module cnn_model_layer1#(

    parameter DATA_W = 16,
    parameter MAC_W  = 32,
    parameter IN_CH  = 1,      // raw audio: 1 channel
    parameter OUT_CH = 16,     // Layer-1: 16 filters
    parameter K_SIZE = 9
)(
    input  wire                 clk,
    input  wire                 reset,

    // serialized raw audio input
    input  wire [DATA_W-1:0]    data_in,
    input  wire                 data_valid_in,

    // serialized output for Layer-2
    output wire [DATA_W-1:0]    pooled_out,
    output wire                 data_valid_out
);

    // ---------------------------------------------------------
    // 1. DESERIALIZER: collect kernel window of size K_SIZE
    //    output: window_vector = [x(t), x(t-1), ... x(t-K+1)]
    // ---------------------------------------------------------
    wire [K_SIZE*DATA_W-1:0] window_vector;
    wire                      window_valid;

    conv1d_window_buffer #(
        .DATA_W(DATA_W),
        .K_SIZE(K_SIZE)
    ) u_window (
        .clk(clk),
        .reset(reset),
        .sample_in(data_in),
        .sample_valid(data_valid_in),
        .window_out(window_vector),
        .window_valid(window_valid)
    );

    // ---------------------------------------------------------
    // 2. MULTI-FILTER CONVOLUTION (IN_CH=1 ? OUT_CH=16)
    // ---------------------------------------------------------
    wire [DATA_W-1:0] conv_out;
    wire              conv_valid;

    multichannel_conv #(
        .DATA_W(DATA_W),
        .MAC_W(MAC_W),
        .IN_CH(K_SIZE),      // because 1×K kernel becomes K inputs
        .OUT_CH(OUT_CH)      // 16 filters
    ) u_conv1 (
        .clk(clk),
        .reset(reset),
        .vector_in(window_vector),
        .vector_valid(window_valid),
        .conv_out(conv_out),
        .conv_valid(conv_valid)
    );

    // ---------------------------------------------------------
    // 3. BATCH NORMALIZATION (per filter output)
    // ---------------------------------------------------------
    wire [DATA_W-1:0] bn_out;
    wire              bn_valid;

    batch_normalization #(
        .DATA_W(DATA_W),
        .MAC_W(MAC_W)
    ) u_bn1 (
        .clk(clk),
        .reset(reset),
        .data_in(conv_out),
        .gamma_in(16'h1000),
        .beta_in(16'h0000),
        .data_valid_in(conv_valid),
        .data_out(bn_out),
        .data_valid_out(bn_valid)
    );

    // ---------------------------------------------------------
    // 4. ReLU
    // ---------------------------------------------------------
    wire [DATA_W-1:0] relu_out;
    wire              relu_valid;

    Relu_fixedPoint #(
        .DATA_W(DATA_W)
    ) u_relu1 (
        .clk(clk),
        .reset(reset),
        .data_in(bn_out),
        .data_valid_in(bn_valid),
        .data_out(relu_out),
        .data_valid_out(relu_valid)
    );

    // ---------------------------------------------------------
    // 5. MAX-POOLING (size = 8 for Layer-1)
    // ---------------------------------------------------------
    max_polling #(
        .DATA_W(DATA_W),
        .POOL_SIZE(8)
    ) u_maxp1 (
        .clk(clk),
        .reset(reset),
        .data_in(relu_out),
        .data_valid_in(relu_valid),
        .data_out(pooled_out),
        .data_valid_out(data_valid_out)
    );

endmodule




`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/21/2025 03:48:37 PM
// Design Name: 
// Module Name: conv1d_window_buffer
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////



module conv1d_window_buffer #(
    parameter DATA_W = 16,
    parameter K_SIZE = 9
)(
    input  wire                 clk,
    input  wire                 reset,
    input  wire [DATA_W-1:0]    sample_in,
    input  wire                 sample_valid,

    output reg [K_SIZE*DATA_W-1:0] window_out,
    output reg                     window_valid
);

    reg [DATA_W-1:0] shift_reg [0:K_SIZE-1];
    integer i;

    always @(posedge clk) begin
        if (reset) begin
            for (i = 0; i < K_SIZE; i = i + 1)
                shift_reg[i] <= 0;
            window_valid <= 0;
        end
        else if (sample_valid) begin
            // shift right
            for (i = K_SIZE-1; i > 0; i = i - 1)
                shift_reg[i] <= shift_reg[i-1];

            shift_reg[0] <= sample_in;

            // produce sliding window
            for (i=0; i < K_SIZE; i=i+1)
                window_out[i*DATA_W +: DATA_W] <= shift_reg[i];

            window_valid <= 1'b1;
        end
        else begin
            window_valid <= 1'b0;
        end
    end
endmodule




`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/21/2025 04:01:37 PM
// Design Name: 
// Module Name: multichannel_conv
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////
// multichannel_conv
// - vector_in: packed [IN_CH * DATA_W - 1 : 0], channel0 at LSB chunk
// - Produces OUT_CH outputs serially (one conv_out per filter), asserting conv_valid
// - Fixed-point fractional bits: FRACT
//////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////
// multichannel_conv  (updated - Vivado compatible)
// ------------------------------------------------
// - vector_in: packed [IN_CH * DATA_W - 1 : 0], ch0 at LSB
// - Computes OUT_CH filter outputs serially
// - conv_valid pulses once per filter output
// - FRACT scaling = 12 bits
//////////////////////////////////////////////////////////////////////////////////

module multichannel_conv #(
    parameter DATA_W = 16,
    parameter MAC_W  = 32,
    parameter IN_CH  = 9,    // For Layer1, IN_CH = K_SIZE (=9)
    parameter OUT_CH = 16    // Number of filters (Layer1 = 16)
)(
    input  wire                     clk,
    input  wire                     reset,

    input  wire [IN_CH*DATA_W-1:0]  vector_in,
    input  wire                     vector_valid,

    output reg  [DATA_W-1:0]        conv_out,
    output reg                      conv_valid
);

    localparam FRACT = 12;

    // ---------------------------------------------------------
    // Index widths
    // ---------------------------------------------------------
    localparam CH_W  = (IN_CH  <= 1) ? 1 : $clog2(IN_CH);
    localparam FIL_W = (OUT_CH <= 1) ? 1 : $clog2(OUT_CH);

    // ---------------------------------------------------------
    // Registers
    // ---------------------------------------------------------
    reg [CH_W-1:0]  ch_idx;
    reg [FIL_W-1:0] f_idx;
    reg             running;

    reg signed [MAC_W-1:0] accumulator;
    reg signed [MAC_W-1:0] next_acc;

    reg [IN_CH*DATA_W-1:0] vec_reg;  // latched input vector

    // ---------------------------------------------------------
    // Weight + bias memory (flattened)
    // ---------------------------------------------------------
    reg signed [DATA_W-1:0] weight_flat [0:IN_CH*OUT_CH-1];
    reg signed [DATA_W-1:0] bias_flat   [0:OUT_CH-1];

    initial begin
        // Optional: load weights for simulation
        // $readmemh("layer1_weights.mem", weight_flat);
        // $readmemh("layer1_biases.mem",  bias_flat);
    end

    // ---------------------------------------------------------
    // Extract current sample/weight and multiply
    // ---------------------------------------------------------
    wire signed [DATA_W-1:0] cur_sample;
    wire signed [DATA_W-1:0] cur_weight;
    wire signed [MAC_W-1:0]  product;

    assign cur_sample = vec_reg[ch_idx*DATA_W +: DATA_W];
    assign cur_weight = weight_flat[f_idx*IN_CH + ch_idx];
    assign product    = cur_sample * cur_weight;

    // ---------------------------------------------------------
    // Scaled result (Vivado-compatible)
    // ---------------------------------------------------------
    wire signed [DATA_W-1:0] scaled_conv;
    assign scaled_conv = (next_acc >>> FRACT);

    // ---------------------------------------------------------
    // MAIN FSM + MAC LOOP
    // ---------------------------------------------------------

    always @(posedge clk) begin
        if (reset) begin
            ch_idx      <= 0;
            f_idx       <= 0;
            running     <= 0;
            accumulator <= 0;
            conv_out    <= 0;
            conv_valid  <= 0;
            vec_reg     <= 0;
        end else begin

            conv_valid <= 1'b0;

            // Start a new computation
            if (vector_valid && !running) begin
                vec_reg     <= vector_in;
                running     <= 1;
                ch_idx      <= 0;
                f_idx       <= 0;
                accumulator <= 0;
            end

            else if (running) begin

                // Compute next_acc
                next_acc = accumulator + product;

                // Last channel ? produce output
                if (ch_idx == IN_CH - 1) begin

                    conv_out   <= scaled_conv + bias_flat[f_idx];
                    conv_valid <= 1'b1;

                    accumulator <= 0;
                    ch_idx <= 0;

                    // Last filter?
                    if (f_idx == OUT_CH - 1) begin
                        f_idx   <= 0;
                        running <= 0;
                    end else begin
                        f_idx <= f_idx + 1'b1;
                    end
                end

                else begin
                    accumulator <= next_acc;
                    ch_idx <= ch_idx + 1'b1;
                end

            end
        end
    end

endmodule


`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/21/2025 03:53:20 PM
// Design Name: 
// Module Name: batch_normalization
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////




module batch_normalization #(
    parameter DATA_W = 16,
    parameter MAC_W  = 32
)(
    input  wire                 clk,
    input  wire                 reset,
    input  wire [DATA_W-1:0]    data_in,
    input  wire [DATA_W-1:0]    gamma_in,
    input  wire [DATA_W-1:0]    beta_in,
    input  wire                 data_valid_in,

    output reg  [DATA_W-1:0]    data_out,
    output reg                  data_valid_out
);

    localparam FRACT = 12;

    wire signed [MAC_W-1:0] prod;
    assign prod = $signed(gamma_in) * $signed(data_in);

    wire signed [DATA_W-1:0] scaled_value;
    assign scaled_value = prod >>> FRACT;

    always @(posedge clk) begin
        if (reset) begin
            data_out       <= 0;
            data_valid_out <= 0;
        end else begin
            data_valid_out <= 0;

            if (data_valid_in) begin
                data_out <= $signed(scaled_value) + $signed(beta_in);
                data_valid_out <= 1'b1;
            end
        end
    end

endmodule



`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/21/2025 03:54:42 PM
// Design Name: 
// Module Name: Relu_fixedPoint
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module Relu_fixedPoint #(
    parameter DATA_W = 16
)(
    input  wire                 clk,
    input  wire                 reset,
    input  wire [DATA_W-1:0]    data_in,
    input  wire                 data_valid_in,
    output reg  [DATA_W-1:0]    data_out,
    output reg                  data_valid_out
);

    always @(posedge clk) begin
        if (reset) begin
            data_out       <= 0;
            data_valid_out <= 0;
        end else begin
            data_valid_out <= 0;

            if (data_valid_in) begin
                // Signed MSB check
                if (data_in[DATA_W-1] == 1'b1)
                    data_out <= 0;
                else
                    data_out <= data_in;

                data_valid_out <= 1'b1;
            end
        end
    end

endmodule



`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/21/2025 03:55:43 PM
// Design Name: 
// Module Name: max_polling
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////



module max_polling #(
    parameter DATA_W   = 16,
    parameter POOL_SIZE = 8
)(
    input  wire                 clk,
    input  wire                 reset,
    input  wire [DATA_W-1:0]    data_in,
    input  wire                 data_valid_in,

    output reg  [DATA_W-1:0]    data_out,
    output reg                  data_valid_out
);

    localparam CTR_W = (POOL_SIZE <= 1) ? 1 : $clog2(POOL_SIZE);

    reg [CTR_W-1:0] counter;
    reg signed [DATA_W-1:0] max_val;

    always @(posedge clk) begin
        if (reset) begin
            counter        <= 0;
            max_val        <= 0;
            data_out       <= 0;
            data_valid_out <= 0;
        end else begin
            data_valid_out <= 0;

            if (data_valid_in) begin
                // First element in pool window
                if (counter == 0) begin
                    max_val <= data_in;
                    counter <= 1;
                end else begin
                    // Compare signed values
                    if ($signed(data_in) > $signed(max_val))
                        max_val <= data_in;

                    // End of pool window
                    if (counter == POOL_SIZE-1) begin
                        data_out       <= max_val;
                        data_valid_out <= 1'b1;
                        counter        <= 0;
                    end else begin
                        counter <= counter + 1'b1;
                    end
                end
            end
        end
    end

endmodule

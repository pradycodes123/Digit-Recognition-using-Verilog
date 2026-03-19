module layer1(
    input clk,
    input start,

    input  signed [15:0] x [0:195],

    output signed [15:0] out [0:31],
    output reg done
);

reg signed [15:0] w [0:32*196-1];
reg signed [15:0] b [0:31];

initial begin
    $readmemh("../weights/w1.mem", w);
    $readmemh("../weights/b1.mem", b);
end


wire signed [15:0] w_split [0:31][0:195];

genvar i, j;

generate
    for (i = 0; i < 32; i = i + 1) begin : split1
        for (j = 0; j < 196; j = j + 1) begin : split2
            assign w_split[i][j] = w[i*196 + j];
        end
    end
endgenerate


wire done_w [0:31];

generate
    for (i = 0; i < 32; i = i + 1) begin : neurons

        neuron #(196) n (
            .clk(clk),
            .start(start),
            .x(x),
            .w(w_split[i]),
            .bias(b[i]),
            .out(out[i]),
            .done(done_w[i])
        );

    end
endgenerate


integer k;

always @(*) begin
    done = 1;
    for (k = 0; k < 32; k = k + 1)
        done = done & done_w[k];
end

endmodule
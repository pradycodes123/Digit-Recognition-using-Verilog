module layer2(
    input clk,
    input start,

    input  signed [15:0] x [0:31],

    output signed [15:0] out [0:15],
    output reg done
);

reg signed [15:0] w [0:16*32-1];
reg signed [15:0] b [0:15];

initial begin
    $readmemh("../weights/w2.mem", w);
    $readmemh("../weights/b2.mem", b);
end


wire signed [15:0] w_split [0:15][0:31];

genvar i, j;

generate
    for (i = 0; i < 16; i = i + 1) begin : split1
        for (j = 0; j < 32; j = j + 1) begin : split2
            assign w_split[i][j] = w[i*32 + j];
        end
    end
endgenerate


wire done_w [0:15];

generate
    for (i = 0; i < 16; i = i + 1) begin : neurons

        neuron #(32) n (
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
    for (k = 0; k < 16; k = k + 1)
        done = done & done_w[k];
end

endmodule
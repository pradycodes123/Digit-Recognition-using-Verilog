module layer3(
    input clk,
    input start,

    input  signed [15:0] x [0:15],

    output signed [15:0] out [0:9],
    output reg done
);

reg signed [15:0] w [0:10*16-1];
reg signed [15:0] b [0:9];

initial begin
    $readmemh("../weights/w3.mem", w);
    $readmemh("../weights/b3.mem", b);
end


wire signed [15:0] w_split [0:9][0:15];

genvar i, j;

generate
    for (i = 0; i < 10; i = i + 1) begin : split1
        for (j = 0; j < 16; j = j + 1) begin : split2
            assign w_split[i][j] = w[i*16 + j];
        end
    end
endgenerate


wire done_w [0:9];

generate
    for (i = 0; i < 10; i = i + 1) begin : neurons

        neuron #(16) n (
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
    for (k = 0; k < 10; k = k + 1)
        done = done & done_w[k];
end

endmodule
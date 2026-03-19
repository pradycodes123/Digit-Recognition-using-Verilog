module mac(
    input signed [15:0] a,
    input signed [15:0] b,
    input signed [31:0] acc_in,
    output signed [31:0] acc_out
);

    assign acc_out = acc_in + (a * b);
endmodule
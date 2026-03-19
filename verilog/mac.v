module mac(
    input clk,
    input signed [15:0] a,
    input signed [15:0] b,
    input signed [31:0] acc_in,
    output reg signed [31:0] acc_out
);

always @(posedge clk) begin
    acc_out <= acc_in + (a * b);
end

endmodule
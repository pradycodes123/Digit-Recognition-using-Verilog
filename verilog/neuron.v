module neuron #(
    parameter N = 196
)(
    input clk,
    input start,

    input  signed [15:0] x [0:N-1],
    input  signed [15:0] w [0:N-1],
    input  signed [15:0] bias,

    output reg signed [15:0] out,
    output reg done
);

integer i;

reg signed [31:0] acc;
wire signed [31:0] mac_out;

reg signed [15:0] a_reg, b_reg;

wire signed [31:0] final_val;

assign final_val = (acc + {{16{bias[15]}}, bias}) >>> 8;


mac m(
    .clk(clk),
    .a(a_reg),
    .b(b_reg),
    .acc_in(acc),
    .acc_out(mac_out)
);


always @(posedge clk) begin

    // ---------- RESET ----------
    if (start) begin
        i    <= 0;
        acc  <= 0;
        done <= 0;
    end


    // ---------- FEED PHASE ----------
    else if (i < N) begin

        // load next inputs into MAC
        a_reg <= x[i];
        b_reg <= w[i];

        // accumulate previous cycle result
        if (i > 0)
            acc <= mac_out;

        i <= i + 1;
    end


    // ---------- DRAIN PHASE ----------
    // capture last MAC result
    else if (i == N) begin
        acc <= mac_out;
        i <= i + 1;
    end


    // ---------- FINISH ----------
    else if (!done) begin

        if (final_val < 0)
            out <= 0;
        else
            out <= final_val[15:0];

        done <= 1;
    end

end

endmodule
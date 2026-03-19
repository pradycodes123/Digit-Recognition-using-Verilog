module nn_top(

    input clk,
    input start,

    input  signed [15:0] x [0:195],

    output signed [15:0] out [0:9],
    output reg done
);


// layer outputs
wire signed [15:0] l1_out [0:31];
wire signed [15:0] l2_out [0:15];
wire signed [15:0] l3_out [0:9];


// done signals
wire done1;
wire done2;
wire done3;


// start signals
reg start1;
reg start2;
reg start3;



// layers

layer1 L1(
    .clk(clk),
    .start(start1),
    .x(x),
    .out(l1_out),
    .done(done1)
);

layer2 L2(
    .clk(clk),
    .start(start2),
    .x(l1_out),
    .out(l2_out),
    .done(done2)
);

layer3 L3(
    .clk(clk),
    .start(start3),
    .x(l2_out),
    .out(l3_out),
    .done(done3)
);


// output connect
assign out = l3_out;



// FSM states

reg [2:0] state;

parameter IDLE = 0;
parameter RUN1 = 1;
parameter RUN2 = 2;
parameter RUN3 = 3;
parameter DONE = 4;


always @(posedge clk) begin

    case(state)

    IDLE: begin
        done <= 0;
        if (start) begin
            start1 <= 1;
            state <= RUN1;
        end
    end


    RUN1: begin
        start1 <= 0;
        if (done1) begin
            start2 <= 1;
            state <= RUN2;
        end
    end


    RUN2: begin
        start2 <= 0;
        if (done2) begin
            start3 <= 1;
            state <= RUN3;
        end
    end


    RUN3: begin
        start3 <= 0;
        if (done3) begin
            state <= DONE;
        end
    end


    DONE: begin
        done <= 1;
    end

    endcase

end


endmodule
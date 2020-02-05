clc; close all;clear all;
an = ann('random',1,5,1);

%1-5-1 p = 5 için
%weights ={[-2.6786   28.4893    6.9116    6.9105   -6.7219],[-85.4954;318.2934;-156.7686;-156.2467;80.2493]};


%1-5-1 p = 10
weights = {[-1.6231   27.6599   21.5158    7.9409   -4.3848],
[
  -46.6677
  318.9777
 -157.7454
 -154.6358
   40.0518
]};



an.set_weights(weights)

%&[in,out] = an.feedforward([1;0]);

%tr_inputs = {[0;0],[0;1],[1;0],[1;1]};
%tr_outputs = {[1],[0],[0],[1]};

num = 10;

in_ = linspace(0,1,num);
out_ = (1+sin(in_*2*pi))*0.5;


tr_inputs = {};
tr_outputs = {};

an.learning_rate = 0.8;

for i = 1:num
	tr_inputs{end+1} = [in_(i)];
	tr_outputs{end+1} = [out_(i)];
end



%tr_inputs = {[1;1]};
%tr_outputs = {[0]};

an.backpropagation(tr_inputs,tr_outputs,-1,0.1);

%[out] = an.feedforward([0;0])
%[out] = an.feedforward([0;1])
%[out] = an.feedforward([1;0])
%[out] = an.feedforward([1;1])

calc_outs = [];
for i = 1:num
	calc_outs(end+1) = an.feedforward(in_(i) );
end


plot(in_,out_,'-o')
hold on
plot(in_,calc_outs,'-o')
legend({'Real','Calc'})
max(abs(out_ - calc_outs))


% ns =

%   Columns 1 through 7

%     4.2244    4.2256    1.2805    4.2246    4.2254   -1.2255    4.2252

%   Columns 8 through 10

%    21.2325    4.2251    1.2800


% ans =

%   -21.9610
%   -24.4823
%    42.2012
%   -22.4129
%   -24.1211
%   -21.3998
%   -23.7755
%    77.4140
%   -23.4677
%    42.0093

% >> 

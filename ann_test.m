clc; close all; clear all;
an = ann('random','bias',1,10,1);


w1= [[ -4.15828466 -23.62272835   9.22304058 -26.40208435  -9.06459713]
 [ -0.31178516  20.25435638  -6.63997412   3.22572756   2.44406056]]
w2= [[ 2.88339257]
 [-6.94020033]
 [-6.96267414]
 [-7.07498121]
 [ 6.00824451]
 [ 6.72118568]]


weights = {w1,w2};


an.set_weights(weights)


num = 10;

in_ = linspace(0,1,num);
out_ = (1+sin(in_*2*pi))*0.5;


tr_inputs = {};
tr_outputs = {};

an.learning_rate =0.1;

for i = 1:num
	tr_inputs{end+1} = [in_(i)];
	tr_outputs{end+1} = [out_(i)];
end



%tr_inputs = {[1;1]};
%tr_outputs = {[0]};

%an.backpropagation(tr_inputs,tr_outputs,-1,0.18);

%[out] = an.feedforward([0;0])
%[out] = an.feedforward([0;1])
%[out] = an.feedforward([1;0])
%[out] = an.feedforward([1;1])

calc_outs = [];
for i = 1:num
	calc_outs(end+1) = an.feedforward(in_(i) );
end

figure(1)
plot(in_,out_,'-o')
hold on
plot(in_,calc_outs,'-o')

x = min(in_):0.05:max(in_);
y = [];
for i = 1:numel(x)
	y(end+1) = an.feedforward(x(i) );
end

plot(x,y)

legend({'Real','Calc','sweep'})
max(abs(out_ - calc_outs));




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

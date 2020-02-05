clc; close all;
an = ann(1,2,2,3,3);

m = 1e-3;

x = linspace(-100*m,100*m,500);
unit = ones(size(x))*m; 
x150 = 150*unit;
x50 = 50*unit;
x0 = 0*unit;

y1 = an.gilbert_multiplier(x0,x);
y2 = an.gilbert_multiplier(x150,x);
y3 = an.gilbert_multiplier(-x150,x);
y4 = an.gilbert_multiplier(x50,x);
y5 = an.gilbert_multiplier(-x50,x);


hold on
plot(x,y1,'-');
plot(x,y2,'-');
plot(x,y3,'-');
plot(x,y4,'-');
plot(x,y5,'-');


%xhes'in ve berilin devresinde k için bir fonksiyon uydurmaya çalış..


clc; close all;clear all;
an = ann('random',1,5,1);

%1-5-1 p = 5 için
%weights ={[-2.6786   28.4893    6.9116    6.9105   -6.7219],[-85.4954;318.2934;-156.7686;-156.2467;80.2493]};


%1-5-1 p = 10
weights = {[-3.6974   28.4527    8.4965    8.0515   -6.1102],
[
  -82.8337
  316.6972
 -158.0688
 -157.5452
   81.6754
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

an.learning_rate = 1.3;

for i = 1:num
	tr_inputs{end+1} = [in_(i)];
	tr_outputs{end+1} = [out_(i)];
end



%tr_inputs = {[1;1]};
%tr_outputs = {[0]};

an.backpropagation(tr_inputs,tr_outputs,-1,0.15);

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

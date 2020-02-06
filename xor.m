clc; close all;clear all;
an = ann('random','bias',2,2,1);

%&[in,out] = an.feedforward([1;0]);

tr_inputs = {[0;0],[0;1],[1;0],[1;1]};
tr_outputs = {[0],[1],[1],[0]};
an.learning_rate = 1.3;


an.backpropagation(tr_inputs,tr_outputs,-1,0.05);

%[out] = an.feedforward([0;0])
%[out] = an.feedforward([0;1])
%[out] = an.feedforward([1;0])
%[out] = an.feedforward([1;1])

calc_outs = [];
for i = 1:numel(tr_inputs)
	calc_outs(end+1) = an.feedforward(tr_inputs{i} );
end

calc_outs


for k = 1:numel(an.weights)
	for i = 1:size(an.weights{k},1)
		for j = 1:size(an.weights{k},2)
			eval(sprintf("w%d_%d%d = %f",k,i,j,an.weights{k}(i,j)))
		end
	end
end
clc; close all;clear all;
an = ann('zeros','bias',2,2,1);

%&[in,out] = an.feedforward([1;0]);

tr_inputs = {[0,0],[0,1],[1,0],[1,1]};
tr_outputs = {[0],[1],[1],[0]};
an.learning_rate = 1.3;


w1= [[-6.87852144  6.03547525]
 [ 6.67953682 -6.33319998]
 [-3.67549896 -3.30782104]]
w2= [[10.62360954]
 [10.74830437]
 [-5.23962641]]



weights = {w1,w2};

an.set_weights(weights);

% an.backpropagation(tr_inputs,tr_outputs,-1,0.05);

[out] = an.feedforward([0,0])
[out] = an.feedforward([0,1])
[out] = an.feedforward([1,0])
[out] = an.feedforward([1,1])

% calc_outs = [];
% for i = 1:numel(tr_inputs)
% 	calc_outs(end+1) = an.feedforward(tr_inputs{i} );
% end

% calc_outs


% for k = 1:numel(an.weights)
% 	for i = 1:size(an.weights{k},1)
% 		for j = 1:size(an.weights{k},2)
% 			eval(sprintf("w%d_%d%d = %f",k,i,j,an.weights{k}(i,j)))
% 		end
% 	end
% end
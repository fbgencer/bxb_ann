classdef ann < handle
   properties
   	no_input = 0;
   	no_output = 0;
   	no_hidden = {0};
    no_neurons = 0;
    no_layer = 2; % total number of layers =  input + hiddens + output 
    weights = {};
    learning_rate = 1;

    best_weights = {};

    is_bias = 0;

   end
   methods
   	    function self = ann(weight_type,is_bias,varargin)
        	self.no_input = varargin{1};
        	self.no_hidden = varargin(2:end-1);
        	self.no_output = varargin{end}; 

          self.no_neurons = varargin;

          if(strcmp(is_bias,"bias"))
            self.is_bias = 1;
            for i = 1:numel(self.no_neurons)-1
              %do not add bias inpt to the output layer
            %self.no_neurons{i} = self.no_neurons{i} + 1;
            end
          end

          self.no_layer = self.no_layer + numel(self.no_hidden); 
          
          
          self.weights = cell(1,self.no_layer-1);

          if(strcmp(weight_type,'random'))
            for i = 1:self.no_layer-1
              self.weights{i} = rand(self.no_neurons{i}+1,self.no_neurons{i+1});
            end
          elseif(strcmp(weight_type,'zeros'))
            for i = 1:self.no_layer-1
              self.weights{i} = zeros(self.no_neurons{i}+1,self.no_neurons{i+1});
            end            
          end
          celldisp(self.weights);
          fprintf("weight sizes :"); disp(self.weights)
        end

        function set_weights(self,w)
          self.weights = w;
        end

        function res = gilbert_multiplier(self,in1,in2)
          if(size(in1) ~= size(in2)), error('Size mismatch'); end
          
          limits = [0,200] * 1e-3;
          saturation_value = 200e-6; % saturation val
          K = 0.089;
          
          for it = 1:numel(in1)
            if( (is_inside(in1(it),limits) || is_inside(in1(it),-limits)) && (is_inside(in2(it),limits) || is_inside(in2(it),-limits)) )
              res(it) = K*in1(it)*in2(it);
            elseif(sign(in1(it))*sign(in2(it)) > 0 )
              res(it) = saturation_value;
            else
              res(it) = -saturation_value;
            end
          end


        end

        function res = vga_multiplier(self,in1,in2)
          limits = [0,700] * 1e-3;
          K = 0.01;

          saturation_value = 500e-6; % saturation val
          
          threshold = 0.25;

          res = 0;

          for it = 1:numel(in1)
            if(in1(it)<=threshold)
              res(it) = 0;
            else
              res(it) = K*in1(it)*in2(it)-K*threshold*in2(it);
              
              if(res(it)>saturation_value)
                res(it) = saturation_value;
              end

              if(res(it) < -saturation_value)
                  res(it) = -saturation_value;
              end
            end
          end
        end

        function res = activation(self,x)
          res = 1./(1+exp(-x));
          %res = tanh(x);
        end

        function res = dx_activation(self,x)
          res = self.activation(x).*(1-self.activation(x));
          %res = 1 - tanh(x).*tanh(x);
        end

        function [inputs,outputs] = feedforward(self,In)
          %input must be a column vector
          inputs = {In};
          outputs = {In};


          if(self.is_bias)
            inputs{1}(:,end+1) = 1;
            outputs{1}(:,end+1) = 1;
          end


          for i = 1:self.no_layer-1
            %weights{i} = zeros(self.no_neurons{i},self.no_neurons{i+1});
            inputs{i+1} = outputs{i}*self.weights{i};
            outputs{i+1} = self.activation(inputs{i+1});

            if(self.is_bias && i ~= self.no_layer-1)
              outputs{i+1}(:,end+1) = 1;
            end

          end



          if(nargout == 1)
            inputs = outputs{end};
          end
        end


        function backpropagation(self,tr_in,tr_out,iteration_number,stop_tolerance)
          %1 tane hidden varsa bu fonksiyon

          delta_weights = cell(1,self.no_layer-1);
          for i = 1:self.no_layer-1
            delta_weights{i} = zeros(self.no_neurons{i},self.no_neurons{i+1});
          end               

          if(numel(tr_in) ~= numel(tr_out))
            error('Training input and output cells must share the same size');
          end

          pattern_no = numel(tr_in);

          fprintf("Starting training with max iteration no %d\n",iteration_number);

          iter = 0;

          tolerance = cell(1,pattern_no);
          best_tolerance = 1e10;

          while(iteration_number == -1 || iter < iteration_number )
            
            dw2 = delta_weights{end};  
            dw1 = delta_weights{end-1};
            
            for pn = 1:pattern_no  
              In = tr_in{pn};
              target = tr_out{pn};

              [inputs,outputs] = self.feedforward(In);

              delta_output = target-outputs{end};
              
              tolerance{pn} = abs(delta_output);

              %fprintf("tol:%f\n",delta_output);

              %celldisp(inputs);
              %celldisp(outputs);
              %celldisp(self.weights)
              %disp('inputs:'); inputs{:}
              %disp('outputs'); outputs{:}
            
              output = outputs{end};
              output_of_hidden = outputs{end-1};
              output_of_input = outputs{end-2};


              input_of_output = inputs{end};
              input_of_hidden = inputs{end-1};

              %from out to hidden
              for j = 1:self.no_output
                for i = 1:self.no_hidden{1}
                  dw2(i,j) = dw2(i,j) + delta_output(j)*self.dx_activation(input_of_output(j))*output_of_hidden(i);
                end
              end
            
              %from hidden to out
              for k = 1:self.no_output
                for j = 1:self.no_hidden{1}
                  for i = 1:self.no_input
                    dw1(i,j) = dw1(i,j) + self.weights{end}(j,k).*delta_output(k).*self.dx_activation(input_of_output(k)).*self.dx_activation(input_of_hidden(j)).*output_of_input(i);
                  end
                end
              end
             
            end

            calc_tol = max(cell2mat(tolerance));
            if(calc_tol <= stop_tolerance)
              fprintf("Tolerance is satisfied. Calculated tolerance :%f\n",calc_tol);
              %celldisp(tolerance)
              break;
            else
              self.weights{end-1} = self.weights{end-1} + self.learning_rate .* dw1;
              self.weights{end} = self.weights{end} + self.learning_rate .* dw2;
            end


            if( calc_tol <= best_tolerance )
              best_tolerance = calc_tol;
              self.best_weights = self.weights;
            end


            iter = iter+1;
            if(mod(iter,10000) == 0)
              fprintf("Iter %d, Tol:%f\n",iter,calc_tol);
            end
          end

          %celldisp(self.weights)





          %return 

          %fprintf('backpropagation finished for iteration_number %d\n',iteration_number);

        end



        %genellestirilmiş durum için bunu kullanalım
        function backpropagation__(self,In,target)

          delta_weights = cell(1,self.no_layer-1);
          for i = 1:self.no_layer-1
            delta_weights{i} = zeros(self.no_neurons{i},self.no_neurons{i+1});
          end          

          outputs = self.feedforward(In);
          outputs{:}
          delta_output = target-outputs{end};

          outputs{end-1}
          %starts with output layer
          delta_weights{end} = (delta_output .* self.dx_activation(outputs{end})) .* transpose(outputs{end-1});
          
          for k = self.no_layer-2:-1:1
            delta_weights{k}(i,j) = delta_output(k)* self.dx_activation(outputs{end})
          end
          
        end
   end
end



function res = is_inside(x,limit_array)

  res = (x >= min(limit_array) ) && (x <= max(limit_array));

end
function [error_train, error_val] = ...
    learningCurve(initial_nn_params,train_x,train_y,test_x,test_y, lambda,...
    input_layer_size,hidden_layer_size1,hidden_layer_size2,num_labels)
% Number of training examples
m = size(train_x, 1);
% dist=20;
% k=m/dist;%�ҵ�ѵ��������200�������ԣ��Ҿ�ѡȡ20��40��.....200�������ֱ����ѵ��
t=3:7;    %ÿ���˷ֱ�ѡ��t�����������в���
k=size(t,2);
% You need to return these values correctly
error_train = zeros(k, 1);
error_val   = zeros(k, 1);
for i=1:k                       %����k��ѵ��
   %���ò���ѵ��������ѵ������theta
   train_x_t=[];
   train_y_t=[];
   dist=t(i);                   %ÿ������dist����ѵ��
   for m=1:40
      for n=1:dist
         a=train_x((m-1)*7+n,:);
         b=train_y((m-1)*7+n,:);
         train_x_t=[train_x_t;a];
         train_y_t=[train_y_t;b];
      end
   end
    theta=train_nn(initial_nn_params,lambda,train_x_t,train_y_t,...
                  input_layer_size,hidden_layer_size1, hidden_layer_size2,num_labels);
%    theta=train_nn(initial_nn_params,lambda,train_x((1+dist*(i-1)):(dist*i),:),train_y((1+dist*(i-1)):(dist*i),:),...
%                   input_layer_size,hidden_layer_size1, hidden_layer_size2,num_labels);
   %You should evaluate the training error on the first i training examples (i.e., X(1:i, :) and y(1:i)).
   %ѵ��������ֻ�ô˵���������Ӧ��ѵ������ train_x((1+20*(i-1)):(20*i)),train_y((1+20*(i-1)):(20*i))
   [error_train(i),grad]=nnCostFunction(theta,input_layer_size,hidden_layer_size1,...
                         hidden_layer_size2,num_labels,train_x_t,train_y_t,0); 
%    [error_train(i),grad]=nnCostFunction(theta,input_layer_size,hidden_layer_size1,...
%                          hidden_layer_size2,num_labels,train_x((1+dist*(i-1)):(dist*i),:),train_y((1+dist*(i-1)):(dist*i),:),0); 
   %������֤�������е���֤������Xval, yval
   [error_val(i),  grad]=nnCostFunction(theta,input_layer_size,hidden_layer_size1,...
                         hidden_layer_size2,num_labels,test_x,test_y,0); 
end
end
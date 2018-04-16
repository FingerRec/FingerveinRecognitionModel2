function test_example_DBN
%%˵����
%0-�������Ĺ��ܣ�ʹ��DBN�㷨��ʵ������ʶ�����ݿ�ʹ��ORL���ݿ⣬�ڵ��������ﵽ3000ʱ��ʶ��׼ȷ��98%����
%1-�˳���ʹ�õ���LBP��Local Binary Pattern����ʵ����������ȡ
%2-�˳��򻹸����˻�ѧϰ���ߵĹ��ܣ����������������������ѵ����������ѧϰ����
%3-DBN���м���RBM���ɣ����������ʵ�ֵ���4�����磬�����-����1-����2-�����
%4-DBN��ѵ�������Ϸ�Ϊ����������RBM��ѵ������ѵ������õ���ʼֵ������ʼ���������磬Ȼ����BP���򴫲��㷨��΢����������
%5-����������ĸ��£�ʹ��matlab�ṩ��fmincg������ǰ��������Ҫ�ȵõ�����Ĵ��ۺ���nnCostFunction
%%
%--------------------------��ORL���ݿ����LBP������ȡ------------------------
mapping=getmapping(8,'u2');%�ȼ���Lbp���ӵ�ӳ���

train_x=[];%%ѵ�����ݼ�
for i=1:40%----------------%ORL���ݿ⹲��40��-------------------------------
    for j=1:7%-------------%ÿ����ѡ��7��������ѵ��-------------------------
        a=imread(strcat('E:\Code\MatlabCode\LBP-DBN-face-recognition-master\LBP-DBN-face-recognition-master\ORL\ORL\s',num2str(i),'_',num2str(j),'.bmp'));
        c=a;
        row=size(c,1);%����ͼƬ������ͼƬ���зֿ飬����4*4�ֿ飬ÿ�����LBP
        col=size(c,2);
        B=mat2cell(c,[row/4 row/4 row/4 row/4],[col/4 col/4 col/4 col/4]);
        H.a=0;        %��ÿ���ӿ����Lbp
        for k=1:16
        H1=lbp(B{k},1,8,mapping,'h'); %LBP histogram in (8,1) neighborhood %using uniform patterns
        H.hist{k}=H1;
        end   
        hist=[H.hist{1},H.hist{2},H.hist{3},H.hist{4},H.hist{5},H.hist{6},H.hist{7},H.hist{8},H.hist{9},H.hist{10},H.hist{11},H.hist{12},H.hist{13},H.hist{14},H.hist{15},H.hist{16}];
        MappedData = mapminmax(hist, 0, 0.5);%���������ݹ�һ����[0,0.5]
        train_x=[train_x;MappedData];
    end
end
train_y=zeros(280,40);%%ѵ�������������ǩ,40*7=280
for i=1:40
    for j=1:7
        train_y((i-1)*7+j,i)=1;
    end
end
%-----------------------------------------------------------------------------
test_x=[];%%�������ݼ�
for i=1:40%----------------%ORL���ݿ⹲��40��-------------------------------
    for j=8:10%------------%ÿ����ѡ��3��������ѵ��-------------------------
        %------------------%E:\My RBM-DBN matlab\ORL\ORL\s�����ݿ��·��----
        a=imread(strcat('E:\Code\MatlabCode\LBP-DBN-face-recognition-master\LBP-DBN-face-recognition-master\ORL\ORL\s',num2str(i),'_',num2str(j),'.bmp'));
        c=a;
        row=size(c,1);%����ͼƬ������ͼƬ���зֿ飬2 
        col=size(c,2);
        B=mat2cell(c,[row/4 row/4 row/4 row/4],[col/4 col/4 col/4 col/4]);
        H.a=0;                       %��ÿ���ӿ����Lbp
        for k=1:16
        H1=lbp(B{k},1,8,mapping,'h'); %LBP histogram in (8,1) neighborhood %using uniform patterns
        H.hist{k}=H1;
        end   
        hist=[H.hist{1},H.hist{2},H.hist{3},H.hist{4},H.hist{5},H.hist{6},H.hist{7},H.hist{8},H.hist{9},H.hist{10},H.hist{11},H.hist{12},H.hist{13},H.hist{14},H.hist{15},H.hist{16}];
        MappedData = mapminmax(hist, 0, 0.5);
        test_x=[test_x;MappedData];
    end
end
test_y=zeros(120,40);%%ѵ����ǩ
for i=1:40
    for j=1:3
        test_y((i-1)*3+j,i)=1;
    end
end
%--------����ת����double��----------------------------------------------------
train_x = double(train_x);
test_x  = double(test_x) ;
train_y = double(train_y);
test_y  = double(test_y);
%%  ex1 train a 100 hidden unit RBM and visualize its weights
% rand('state',0)
% dbn.sizes = [100];
% % opts.numepochs =   1;
% opts.numepochs =5;
% % opts.batchsize = 100;
% opts.batchsize = 1;
% opts.momentum  =   0;
% opts.alpha     =   1;
% dbn = dbnsetup(dbn, train_x, opts);
% dbn = dbntrain(dbn, train_x, opts);
% figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
%---------------����DBN����ĸ��ֲ���---------------------------------------
rand('state',0)
%train dbn
dbn.sizes = [100 100];%DBN������������100-100
opts.numepochs = 30;   %��������Ϊ30
opts.batchsize = 1;   %ÿ�δ���batchsize������
opts.momentum  =   0;
opts.alpha     =    0.001;%ѧϰ��Ϊ0.01
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
%%
%--------------------DBN���������ʼ��NN-----------------------------------
input_layer_size  = size(train_x,2);  %�������ݵ�ά������Ӧ�ɼ������Ŀ
hidden_layer_size1=100;
hidden_layer_size2=100;
num_labels=40;                        %���ԣ�����4��������input_layer_size-100-100-40
% Theta1=randInitializeWeights(input_layer_size,hidden_layer_size1);
% Theta2=randInitializeWeights(hidden_layer_size1,hidden_layer_size2);
Theta1=[dbn.rbm{1}.c dbn.rbm{1}.W];   %ѵ���õ�DBN��������ʼ��������
Theta2=[dbn.rbm{2}.c dbn.rbm{2}.W];   %
Theta3=randInitializeWeights(hidden_layer_size2,num_labels);%���������������ʼ��
initial_nn_params = [Theta1(:) ; Theta2(:);Theta3(:)];
lambda = 0.00003;%���򻯲������ƹ����
%%
%--------------------------ѵ��������-------------------------------------
nn_params=train_nn(initial_nn_params,lambda,train_x,train_y,...
          input_layer_size,hidden_layer_size1, hidden_layer_size2,num_labels);
%%
%------------------------��ѵ������NN����Ԥ������ܲ���--------------------
%����������ԭ
Theta1 = reshape(nn_params(1:hidden_layer_size1 * (input_layer_size + 1)), ...
                 hidden_layer_size1, (input_layer_size + 1));
 
first=1+hidden_layer_size1 * (input_layer_size + 1);
second=hidden_layer_size1 * (input_layer_size + 1)+hidden_layer_size2 * (hidden_layer_size1+ 1);
Theta2 = reshape(nn_params(first:second), ...
                 hidden_layer_size2, (hidden_layer_size1 + 1));
                 
first=1+hidden_layer_size1 * (input_layer_size + 1)+hidden_layer_size2 * (hidden_layer_size1+ 1);     
Theta3 = reshape(nn_params(first:end), ...
                 num_labels, (hidden_layer_size2 + 1));

%����Ԥ��
pred = predict(Theta1, Theta2,Theta3, test_x);
%������ȷ��
[dummy, expected] = max(test_y,[],2);
 bad = find(pred ~= expected);    
 er = numel(bad) / size(test_x, 1);
 fprintf('������ȷ����%.2f',1 - er);
 assert(er < 0.10, 'Too big error');
%%
%--------------����������������ѧϰ����----------------------------------
% lambda = 0.03;%���򻯲���
% [hidden_node, error_train, error_val] =hidden_node_learn_curve(lambda,train_x,train_y,test_x,test_y);
% figure(2);
% plot(hidden_node, error_train, hidden_node, error_val);
% legend('Train', 'Cross Validation');
% xlabel('hidden node');
% ylabel('Error');
% axis([100 200 0 1])
%--------------------------------------------------------------------------
%%
%------------------������ѵ��������ģ��ѧϰ����----------------------------
% % lambda =1;%���򻯲���
% lambda = 0.03;%���򻯲���
% figure(1);
% % m=size(train_x,1)/20; %m=20,��20����
% m=3:7;
% [error_train, error_val] = ...
%     learningCurve(initial_nn_params,train_x,train_y,test_x,test_y, lambda,...
%     input_layer_size,hidden_layer_size1,hidden_layer_size2,num_labels);
% plot(m, error_train, m, error_val);
% 
% title(sprintf('Learning Curve (lambda = %f)', lambda));
% xlabel('Number of training examples')
% ylabel('Error')
% axis([3 7 0 5])
% legend('Train', 'Cross Validation')

%--------------------------------------------------------------------------
%%
%---------------���������򻯲���lambda��ѧϰ����---------------------------
% [lambda_vec, error_train, error_val] = ...
%     validationCurve(initial_nn_params,train_x,train_y,test_x,test_y,...
%     input_layer_size,hidden_layer_size1,hidden_layer_size2,num_labels);
% 
% close all;
% figure(2);
% plot(lambda_vec, error_train, lambda_vec, error_val);
% legend('Train', 'Cross Validation');
% xlabel('lambda');
% ylabel('Error');
% axis([0 0.04 0 10])
%---------------------------------------------------------------------------

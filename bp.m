function bp_fingervein
%实现bp神经网络训练数据
%先用lbp提取特征向量，再进行训练
%首先进行预处理

%12张图片用来训练
workspace;
mapping = getmapping(8,'u2'); % 使用正常lbp模式，涉及到旋转问题再解决
size_layer1_all = 50;%总的nn的第一层隐藏层的数目
size_layer2_all = 50;%总的nn的第二层隐藏层的数目
%mapping = getmapping(8,'ri')
train_x = []; %训练输入
for i = 1 :64
    for j = 1 : 12
        if(i < 10 && j < 10)
            img = imread(strcat('E:\图片数据库\617574\FV_samples\F0',num2str(i),'0',num2str(j),'.bmp'));
        elseif(i < 10 && j >= 10)
            img = imread(strcat('E:\图片数据库\617574\FV_samples\F0',num2str(i),num2str(j),'.bmp'));
        elseif(i >= 10 && j < 10)
            img = imread(strcat('E:\图片数据库\617574\FV_samples\F',num2str(i),'0',num2str(j),'.bmp'));    
        elseif(i >= 10 && j >= 10)
            img = imread(strcat('E:\图片数据库\617574\FV_samples\F',num2str(i),num2str(j),'.bmp'));    
        end
        
        preimg = preprocess(img);
        [row col] = size(preimg);
        divideimg = mat2cell(preimg,[row/4 row/4 row/4 row/4],[col/4 col/4 col/4 col/4]);%分为16x16的小块
        %拼接为长度16的特征向量
        for k = 1 : 16
            histimg = lbp(divideimg{k},1,8,mapping,'h'); %得到灰度直方图
            h{k} = histimg; 
        end
        %fprintf('Histimg size is :');
        %size(histimg)
        %fprintf('\n');
        hist = [h{1},h{2},h{3},h{4},h{5},h{6},h{7},h{8},h{9},h{10},h{11},h{12},h{13},h{14},h{15},h{16}];
        %fprintf('Hist size is :');
        %size(hist)
        %fprintf('\n');
        mapdata = mapminmax(hist,0,0.5); % 数据归一化到0到0.5
        train_x = [train_x;mapdata];
    end
end
%size(train_x)
train_y = zeros(768,64); % 标签，64 * 12 = 768
%贴标签
for i = 1 : 64
    for j = 1 : 12
        train_y((i - 1) * 12 + j,i) = 1;
    end
end
%3张图片用来验证
test_x = [];
for i = 1 : 64
    for j = 13 : 15
        if(i < 10 && j < 10)
            img = imread(strcat('E:\图片数据库\617574\FV_samples\F0',num2str(i),'0',num2str(j),'.bmp'));
        elseif(i < 10 && j >= 10)
            img = imread(strcat('E:\图片数据库\617574\FV_samples\F0',num2str(i),num2str(j),'.bmp'));
        elseif(i >= 10 && j < 10)
            img = imread(strcat('E:\图片数据库\617574\FV_samples\F',num2str(i),'0',num2str(j),'.bmp'));    
        elseif(i >= 10 && j >= 10)
            img = imread(strcat('E:\图片数据库\617574\FV_samples\F',num2str(i),num2str(j),'.bmp'));    
        end
        preimg = preprocess(img);
        [row col] = size(preimg);
        divideimg = mat2cell(preimg,[row/4 row/4 row/4 row/4],[col/4 col/4 col/4 col/4]);%分为16x16的小块
        %拼接为长度16的特征向量
        for k = 1 : 16
            histimg = lbp(divideimg{k},1,8,mapping,'h'); %得到灰度直方图
            h{k} = histimg; 
        end
        hist = [h{1},h{2},h{3},h{4},h{5},h{6},h{7},h{8},h{9},h{10},h{11},h{12},h{13},h{14},h{15},h{16}];
        mapdata = mapminmax(hist,0,0.5); % 数据归一化到0到0.5
        test_x = [test_x;mapdata];
    end
end
test_y = zeros(192,64); % 测试标签，64 * 3 = 192
%贴标签
for i = 1 : 64
    for j = 1 : 3
        test_y((i - 1) * 3 + j,i) = 1;
    end 
end
train_x = double(train_x);
test_x =  double(test_x);
train_y = double(train_y);
test_y = double(test_y);

rand('state',0)
%train dbn
dbn.sizes = [size_layer1_all size_layer2_all];%DBN的两个隐层数目
opts.numepochs = 30;   %迭代次数为30
opts.batchsize = 1;   %每次处理batchsize个数据
opts.momentum  =   0;
opts.alpha     =    0.001;%学习率为0.001，迭代下降速度
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts); %RBM每一层搭建DBN，受限波尔慈曼机
% dbn的安装与训练，由多层RBM组成，训练好的dbn初始化神经网络


%实现dbn工具箱的dbnunfoldtonn的功能，吧参数传递给大的NN
%--------------------DBN网络参数初始化NN-----------------------------------
input_layer_size  = size(train_x,2);  %输入数据的维数，对应可见层的数目
hidden_layer_size1=size_layer1_all;
hidden_layer_size2=size_layer2_all;
num_labels = 64;                        %所以，整个4层网络是input_layer_size-size1-size2-64
Theta1=[dbn.rbm{1}.c dbn.rbm{1}.W];   %训练好的DBN参数来初始化神经网络
Theta2=[dbn.rbm{2}.c dbn.rbm{2}.W];   %
Theta3=randInitializeWeights(hidden_layer_size2,num_labels);%最后输出层用随机初始化
initial_nn_params = [Theta1(:) ; Theta2(:);Theta3(:)];
lambda = 0.00003;%正则化参数抑制过拟合

%大NN的训练
%--------------------------训练神经网络-------------------------------------
nn_params=train_nn(initial_nn_params,lambda,train_x,train_y,...
          input_layer_size,hidden_layer_size1, hidden_layer_size2,num_labels);      
%保存训练好的网络
%save('mynn','nn_params');
%------------------------对训练完后的NN进行预测和性能测试--------------------
%将参数矩阵还原
Theta1 = reshape(nn_params(1:hidden_layer_size1 * (input_layer_size + 1)), ...
                 hidden_layer_size1, (input_layer_size + 1));
 
first=1+hidden_layer_size1 * (input_layer_size + 1);
second=hidden_layer_size1 * (input_layer_size + 1)+hidden_layer_size2 * (hidden_layer_size1+ 1);
Theta2 = reshape(nn_params(first:second), ...
                 hidden_layer_size2, (hidden_layer_size1 + 1));
                 
first=1+hidden_layer_size1 * (input_layer_size + 1)+hidden_layer_size2 * (hidden_layer_size1+ 1);     
Theta3 = reshape(nn_params(first:end), ...
                 num_labels, (hidden_layer_size2 + 1));

%进行预测，网络间的传递函数为激活函数
pred = predict(Theta1, Theta2,Theta3, test_x);
%计算正确率
%在第二维度上操作，确保符合测试样本
[dummy, expected] = max(test_y,[],2);
 %size(pred)
 %size(dummy)
 %size(test_y)
 %size(expected)
 length = size(pred,1);
 for i = 1 : length
     fprintf('Predicted User Is：%d;Real User is：%d\n',pred(i),expected(i));
 end
 figure;
 scatter(pred,expected);
 
 bad = find(pred ~= expected);    
 er = numel(bad) / size(test_x, 1);
 fprintf('Correct Rate Is%.5f',1 - er);
 assert(er < 0.10, 'Too big error');
 %若不满足er<0.1,提示错误太大
 
 %
% %------------------描绘关于训练样本规模的学习曲线----------------------------
% % lambda =1;%正则化参数
% lambda = 0.03;%正则化参数
% figure(1);
% % m=size(train_x,1)/20; %m=20,描20个点
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
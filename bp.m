function bp_fingervein
%ʵ��bp������ѵ������
%����lbp��ȡ�����������ٽ���ѵ��
%���Ƚ���Ԥ����

%12��ͼƬ����ѵ��
workspace;
mapping = getmapping(8,'u2'); % ʹ������lbpģʽ���漰����ת�����ٽ��
size_layer1_all = 50;%�ܵ�nn�ĵ�һ�����ز����Ŀ
size_layer2_all = 50;%�ܵ�nn�ĵڶ������ز����Ŀ
%mapping = getmapping(8,'ri')
train_x = []; %ѵ������
for i = 1 :64
    for j = 1 : 12
        if(i < 10 && j < 10)
            img = imread(strcat('E:\ͼƬ���ݿ�\617574\FV_samples\F0',num2str(i),'0',num2str(j),'.bmp'));
        elseif(i < 10 && j >= 10)
            img = imread(strcat('E:\ͼƬ���ݿ�\617574\FV_samples\F0',num2str(i),num2str(j),'.bmp'));
        elseif(i >= 10 && j < 10)
            img = imread(strcat('E:\ͼƬ���ݿ�\617574\FV_samples\F',num2str(i),'0',num2str(j),'.bmp'));    
        elseif(i >= 10 && j >= 10)
            img = imread(strcat('E:\ͼƬ���ݿ�\617574\FV_samples\F',num2str(i),num2str(j),'.bmp'));    
        end
        
        preimg = preprocess(img);
        [row col] = size(preimg);
        divideimg = mat2cell(preimg,[row/4 row/4 row/4 row/4],[col/4 col/4 col/4 col/4]);%��Ϊ16x16��С��
        %ƴ��Ϊ����16����������
        for k = 1 : 16
            histimg = lbp(divideimg{k},1,8,mapping,'h'); %�õ��Ҷ�ֱ��ͼ
            h{k} = histimg; 
        end
        %fprintf('Histimg size is :');
        %size(histimg)
        %fprintf('\n');
        hist = [h{1},h{2},h{3},h{4},h{5},h{6},h{7},h{8},h{9},h{10},h{11},h{12},h{13},h{14},h{15},h{16}];
        %fprintf('Hist size is :');
        %size(hist)
        %fprintf('\n');
        mapdata = mapminmax(hist,0,0.5); % ���ݹ�һ����0��0.5
        train_x = [train_x;mapdata];
    end
end
%size(train_x)
train_y = zeros(768,64); % ��ǩ��64 * 12 = 768
%����ǩ
for i = 1 : 64
    for j = 1 : 12
        train_y((i - 1) * 12 + j,i) = 1;
    end
end
%3��ͼƬ������֤
test_x = [];
for i = 1 : 64
    for j = 13 : 15
        if(i < 10 && j < 10)
            img = imread(strcat('E:\ͼƬ���ݿ�\617574\FV_samples\F0',num2str(i),'0',num2str(j),'.bmp'));
        elseif(i < 10 && j >= 10)
            img = imread(strcat('E:\ͼƬ���ݿ�\617574\FV_samples\F0',num2str(i),num2str(j),'.bmp'));
        elseif(i >= 10 && j < 10)
            img = imread(strcat('E:\ͼƬ���ݿ�\617574\FV_samples\F',num2str(i),'0',num2str(j),'.bmp'));    
        elseif(i >= 10 && j >= 10)
            img = imread(strcat('E:\ͼƬ���ݿ�\617574\FV_samples\F',num2str(i),num2str(j),'.bmp'));    
        end
        preimg = preprocess(img);
        [row col] = size(preimg);
        divideimg = mat2cell(preimg,[row/4 row/4 row/4 row/4],[col/4 col/4 col/4 col/4]);%��Ϊ16x16��С��
        %ƴ��Ϊ����16����������
        for k = 1 : 16
            histimg = lbp(divideimg{k},1,8,mapping,'h'); %�õ��Ҷ�ֱ��ͼ
            h{k} = histimg; 
        end
        hist = [h{1},h{2},h{3},h{4},h{5},h{6},h{7},h{8},h{9},h{10},h{11},h{12},h{13},h{14},h{15},h{16}];
        mapdata = mapminmax(hist,0,0.5); % ���ݹ�һ����0��0.5
        test_x = [test_x;mapdata];
    end
end
test_y = zeros(192,64); % ���Ա�ǩ��64 * 3 = 192
%����ǩ
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
dbn.sizes = [size_layer1_all size_layer2_all];%DBN������������Ŀ
opts.numepochs = 30;   %��������Ϊ30
opts.batchsize = 1;   %ÿ�δ���batchsize������
opts.momentum  =   0;
opts.alpha     =    0.001;%ѧϰ��Ϊ0.001�������½��ٶ�
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts); %RBMÿһ��DBN�����޲���������
% dbn�İ�װ��ѵ�����ɶ��RBM��ɣ�ѵ���õ�dbn��ʼ��������


%ʵ��dbn�������dbnunfoldtonn�Ĺ��ܣ��ɲ������ݸ����NN
%--------------------DBN���������ʼ��NN-----------------------------------
input_layer_size  = size(train_x,2);  %�������ݵ�ά������Ӧ�ɼ������Ŀ
hidden_layer_size1=size_layer1_all;
hidden_layer_size2=size_layer2_all;
num_labels = 64;                        %���ԣ�����4��������input_layer_size-size1-size2-64
Theta1=[dbn.rbm{1}.c dbn.rbm{1}.W];   %ѵ���õ�DBN��������ʼ��������
Theta2=[dbn.rbm{2}.c dbn.rbm{2}.W];   %
Theta3=randInitializeWeights(hidden_layer_size2,num_labels);%���������������ʼ��
initial_nn_params = [Theta1(:) ; Theta2(:);Theta3(:)];
lambda = 0.00003;%���򻯲������ƹ����

%��NN��ѵ��
%--------------------------ѵ��������-------------------------------------
nn_params=train_nn(initial_nn_params,lambda,train_x,train_y,...
          input_layer_size,hidden_layer_size1, hidden_layer_size2,num_labels);      
%����ѵ���õ�����
%save('mynn','nn_params');
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

%����Ԥ�⣬�����Ĵ��ݺ���Ϊ�����
pred = predict(Theta1, Theta2,Theta3, test_x);
%������ȷ��
%�ڵڶ�ά���ϲ�����ȷ�����ϲ�������
[dummy, expected] = max(test_y,[],2);
 %size(pred)
 %size(dummy)
 %size(test_y)
 %size(expected)
 length = size(pred,1);
 for i = 1 : length
     fprintf('Predicted User Is��%d;Real User is��%d\n',pred(i),expected(i));
 end
 figure;
 scatter(pred,expected);
 
 bad = find(pred ~= expected);    
 er = numel(bad) / size(test_x, 1);
 fprintf('Correct Rate Is%.5f',1 - er);
 assert(er < 0.10, 'Too big error');
 %��������er<0.1,��ʾ����̫��
 
 %
% %------------------������ѵ��������ģ��ѧϰ����----------------------------
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
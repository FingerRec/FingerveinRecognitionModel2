i = 1;
j = 1;
a = imread(strcat('E:\ͼƬ���ݿ�\617574\FV_samples\F0',num2str(i),'0',num2str(j),'.bmp'));
b = preprocess(a);
imshow(b);
% mapping=getmapping(8,'u2'); 
% c = lbp(a,1,8,mapping,'h');
% %�뾶Ϊ1��Բ��8����
% subplot(2,1,1),stem(c);
% 
% i =3;
% j = 9;
% b = imread(strcat('E:\ͼƬ���ݿ�\617574\FV_samples\F0',num2str(i),'0',num2str(j),'.bmp'));
% %h2 = lbp(a);
% mapping=getmapping(8,'u2'); 
% c = lbp(b,1,8,mapping,'h');
% subplot(2,1,2),stem(c);

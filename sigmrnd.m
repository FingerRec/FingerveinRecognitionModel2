function X = sigmrnd(P)
%�ڽ�Matlab�ĳ�����ֲ��C++��ʱ�����ʺϲ�������������У���[0,1]֮�������������Ǿ���[1/3]�����
%     X = double(1./(1+exp(-P)))+1*randn(size(P));
%     A=ones(size(P));
%     B=A/3;
%     C=1./(1+exp(-P));
%     X = double(1./(1+exp(-P)) > B);
    X = double(1./(1+exp(-P)) > rand(size(P)));
end
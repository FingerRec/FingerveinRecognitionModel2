function dbn = dbntrain(dbn, x, opts)
    n = numel(dbn.rbm);
    dbn.rbm{1} = rbmtrain(dbn.rbm{1}, x, opts);
    %DBN���ɶ��RBM��ɣ�ѵ�����������ѵ������ѵ����һ�����磬Ȼ��̶���һ������Ĳ���������һ������������Ϊ��һ�����������
    for i = 2 : n
        x = rbmup(dbn.rbm{i - 1}, x);
        opts.numepochs=opts.numepochs;        
        dbn.rbm{i} = rbmtrain(dbn.rbm{i}, x, opts);
    end
end

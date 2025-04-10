clc; clear;
addpath(genpath('./'))
str1={'DD'};

m = length(str1);  
lambda = 10;
alpha = 0.1;
beta = 0.1;   

rng('default'); %随机数种子
for i = 1:m   %控制数据集的个数
    filename = [str1{i} 'Train.mat'];
    load (filename); 
    [X,Y,Z]=create_SubTable2(data_array, tree);   %利用中间结点构造子树，并以此将分类任务拆分为多个子任务
    [feature{i},W{i}] = HFS_instance_label(X, Y, tree, lambda,alpha,beta, 1);   %选择特征子集

    %Test feature batch
    testFile = [str1{i}, 'Test.mat'];
    load (testFile);

    [SVM_accuracyMean{i}, SVM_accuracyStd{i}, SVM_F_LCAMean{i}, SVM_FHMean{i}, SVM_TIEmean{i}, SVM_TestTime{i}] = HierSVMPredictionBatchall1(data_array, tree, feature{i},str1{i});
    [t_r,~]=size(data_array);
    SVM_tiemean{i}=SVM_TIEmean{i}/t_r;

    cd('result\')
    filename=['result',str1{i},'.mat'];
    save(filename,"feature","W","str1",'SVM_accuracyMean',"SVM_accuracyStd","SVM_F_LCAMean","SVM_FHMean","SVM_tiemean");
    cd('..\')

end
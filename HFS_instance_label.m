%% HFS_L21
%最小二乘损失
%L-21稀疏项
%标记
% 样本相关性
function [feature_slct,W] = HFS_instance_label(X,Y,tree,lambda,alpha,beta,flag)

rand('seed',1);
internalNodes = tree_InternalNodes(tree);
indexRoot = tree_Root(tree);% The root of the tree
[~,d] = size(X{indexRoot}); % get the number of features
%noLeafNode =[internalNodes;indexRoot];
eps = 1e-8; % set your own tolerance
maxIte = 10;
noi_nl = [];
for noi = 1:length(internalNodes)        %%针对CLEF数据集的 
    if isempty(Y{internalNodes(noi)})
        noi_nl = [noi_nl, noi];
        W{internalNodes(noi)} = rand(d, length(get_children_set(tree, noi)));
    end
end
nosamplenode=internalNodes(noi_nl);
internalNodes(noi_nl) = [];
noLeafNode =[internalNodes;indexRoot];

for i = 1:length(noLeafNode)   %计算每个子任务中的标记数量
    ClassLabel = unique(Y{noLeafNode(i)});
    m(noLeafNode(i)) = length(ClassLabel);   %m存储每个中间结点对应的子标签的数量
end
maxm=max(m);     %便于统一所有中间几点对应W的列数

for i = 1:length(noLeafNode)   %根据标记数量将每个子任务的标记转换为0,1变量
    Y{noLeafNode(i)}=conversionY01_extendlhy(Y{noLeafNode(i)},m(noLeafNode(i)),noLeafNode(i),tree);%extend 2 to [1 0]
    [r,c] = size(Y{noLeafNode(i)});
    Ytemp = zeros(r,maxm-c);
    Y{noLeafNode(i)}=[Y{noLeafNode(i)} Ytemp];
end

%% 得到任意两个中间结点的路径
for ii = 1:length(noLeafNode)
    for jj = 1:length(noLeafNode)
        nodes_path = Nodepath(noLeafNode(ii),noLeafNode(jj),tree);
        Paths(noLeafNode(ii),noLeafNode(jj)) = nodes_path;  %表示中间结点之间路径
    end
end

%% instance correlation 
for j=1:length(noLeafNode)
    S{noLeafNode(j)}=ins_similarity3(X{noLeafNode(j)},10);
    L1{noLeafNode(j)} = diag(sum(S{noLeafNode(j)},2)) - S{noLeafNode(j)};
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k=1;
[~,d] = size(X{indexRoot}); % get the number of features

%% initialize
for j = 1:length(noLeafNode)
%     W{noLeafNode(j)} = rand(d,m(noLeafNode(j)));
    W{noLeafNode(j)} = rand(d, maxm); % initialize W

    XX{noLeafNode(j)} = X{noLeafNode(j)}' * X{noLeafNode(j)};
    XY{noLeafNode(j)} = X{noLeafNode(j)}' * Y{noLeafNode(j)};
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:maxIte
    %% L21
    for j = 1:length(noLeafNode)
        D{noLeafNode(j)} = diag(0.5./max(sqrt(sum(W{noLeafNode(j)}.*W{noLeafNode(j)},2)),eps));  %L2,1中的拉普拉斯矩阵Li
    end

    %% Update the root node
    W_curent = zeros(d,maxm);
    for j=1:length(noLeafNode)
        if isempty(W{noLeafNode(j)})
            continue
        end
        W_curent = W_curent + Paths(indexRoot,noLeafNode(j))*W{noLeafNode(j)};
    end
    W{indexRoot} = lsqminnorm((XX{indexRoot} + lambda * D{indexRoot} + beta*X{indexRoot}'*L1{indexRoot}*X{indexRoot}),(XY{indexRoot}-alpha*W_curent));
    %W{indexRoot} = inv(XX{indexRoot} + 2 * lambda * D{indexRoot}) * (XY{indexRoot}-alpha*W_current);

    %% Update the internal nodes
    for j = 1:length(internalNodes)
        W_curent1 = zeros(d,maxm);
        compareNode1 = setdiff(noLeafNode,internalNodes(j));  %每个节点需要比较的值
        for k = 1:length(compareNode1)
            if isempty(W{compareNode1(k)})
                continue
            end
            W_curent1 = W_curent1+Paths(internalNodes(j),compareNode1(k))*W{compareNode1(k)};
        end
        W{internalNodes(j)} = lsqminnorm((XX{internalNodes(j)} + lambda * D{internalNodes(j)} + beta*X{internalNodes(j)}'*L1{internalNodes(j)}*X{internalNodes(j)}),(XY{internalNodes(j)}-alpha*W_curent1));
        %W{internalNodes(j)} = inv(XX{internalNodes(j)} + 2 * lambda * D{internalNodes(j)}) * (XY{internalNodes(j)}-alpha*W_curent1);
    end

    %% The value of object function
    if (flag == 1)
        W_curent=0;
        obj(i) = norm(X{indexRoot}*W{indexRoot}-Y{indexRoot},'fro')^2 + lambda * L21(W{indexRoot})+beta*trace((X{indexRoot}*W{indexRoot})'*L1{indexRoot}*(X{indexRoot}*W{indexRoot}));
        for j = 1:length(noLeafNode)
            if isempty(W{noLeafNode(j)})
                continue
            end
            W_curent = W_curent + Paths(indexRoot,noLeafNode(j))*(trace(W{indexRoot}'*W{noLeafNode(j)})+trace(W{indexRoot}*W{noLeafNode(j)}'));
        end
        obj(i) = obj(i) + alpha*W_curent;
        for j = 1:length(internalNodes)
            W_curent1 = 0;
            compareNode1 = setdiff(noLeafNode,internalNodes(j));  %每个节点需要比较的值
            for k = 1:length(compareNode1)
                if isempty(W{compareNode1(k)})
                    continue
                end
                W_curent1 = W_curent1+Paths(internalNodes(j),compareNode1(k)) * (trace(W{internalNodes(j)}'*W{compareNode1(k)})+trace(W{internalNodes(j)}*W{compareNode1(k)}'));
            end
            obj(i) = obj(i) + norm(X{internalNodes(j)}*W{internalNodes(j)} - Y{internalNodes(j)},'fro')^2 + lambda * L21(W{internalNodes(j)})+beta*trace((X{internalNodes(j)}*W{internalNodes(j)})'*L1{internalNodes(j)}*(X{internalNodes(j)}*W{internalNodes(j)}))+alpha*W_curent1;
        end
    end
end
noLeafNode=[noLeafNode;nosamplenode];
for j = 1: length(noLeafNode)
    tempVector = sum(W{noLeafNode(j)}.^2, 2);
    [atemp, value] = sort(tempVector, 'descend'); % sort tempVecror (W) in a descend order
    clear tempVector;
    feature_slct{noLeafNode(j)} = value(1:end);
end

if (flag == 1)
    figure;
    set(gcf,'color','w');
    plot(obj,'LineWidth',4,'Color',[0 0 1]);
    set(gca,'FontName','Times New Roman','FontSize',18);
    xlabel('Iteration number');
    ylabel('Objective function value');
end
end


function [ output_args ] = conversionY01_extendlhy( Y, numCol , cur_node,tree)
    gnd = Y;
    %ClassLabel = get_children_set(tree, cur_node); %Count the number of labels
    ClassLabel = unique(Y);
    nClass = length(ClassLabel);    
    [nSmp,~] = size(Y);
    Y = eye(nClass,nClass);
    Z = zeros(nSmp,numCol);
    for i=1:nClass
        idx = find(gnd==ClassLabel(i));
        Z(idx,1:nClass) = repmat(Y(i,1:nClass),length(idx),1);
    end    
    output_args= Z;
end


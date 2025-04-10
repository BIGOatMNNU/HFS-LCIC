%热核函数
function S = ins_similarity3(X,K)
    A     = pdist2(X,X);
    [num_dim,~] = size(A);
    ss = (sum(sum(A,2)))/(num_dim*(num_dim-1));
    if num_dim>=10
        for i =1:num_dim
            temp = A(i,:);
            As =sort(temp);
            temp  = (temp <=  As(K));
            index = find(temp==1);
            temp1 = zeros(1,num_dim);
            for j = 1:length(index)
                temp1(1,index(j)) = exp(-(A(i,index(j))^2)/(2*ss^2));
            end
            A(i,:) = temp1;
        end
    else
        for i=1:num_dim
            temp = A(i,:);
            As =sort(temp);
            temp  = (temp <=  As(num_dim));
            index = find(temp==1);
            temp1 = zeros(1,num_dim);
            for j = 1:length(index)
                temp1(1,index(j)) = exp(-(A(i,index(j))^2)/(2*ss^2));
            end
            A(i,:) = temp1;
        end
    end
    S = A;
end


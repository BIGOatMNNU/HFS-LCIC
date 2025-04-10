%% 求node1与node2的最近公共祖先结点
function [LCA_node] = tree_LCA(node1,node2,tree)
%%先得到node1,node2的祖先节点
anc1 = tree_Ancestor(tree,node1);
anc2 = tree_Ancestor(tree,node2);

new_anc1 = union(anc1,node1);
new_anc2 = union(anc2,node2);

c_LCA = intersect(new_anc1,new_anc2);

if length(c_LCA) == 1
    LCA_node = c_LCA;
else
    [~,index] = max(tree(c_LCA,2));
    LCA_node = c_LCA(index);
end

end


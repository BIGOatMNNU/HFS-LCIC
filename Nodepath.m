%%用于计算树中两节点间的路径距离
function [nodes_path] = Nodepath(node1,node2,tree)
%%node1 表示第一个结点
%%node2 表示第二个结点

%root_node = tree_Root( tree );

anc1 = tree_Ancestor(tree,node1);
anc2 = tree_Ancestor(tree,node2);

dist1 = length(anc1);
dist2 = length(anc2);

LCA_node = tree_LCA(node1,node2,tree);
anc3 = tree_Ancestor(tree,LCA_node);
dist3 = length(anc3);

nodes_path = dist1+dist2-2*dist3;

end


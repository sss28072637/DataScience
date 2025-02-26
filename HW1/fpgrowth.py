from collections import defaultdict

class TreeNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.next = None
    
    def increment_count(self, count):
        self.count += count

def create_fptree(dataset, min_support):
    header_table = defaultdict(int)
    for transaction in dataset:
        for item in transaction:
            header_table[item] += 1
    
    header_table = {k: v for k, v in header_table.items() if v >= min_support}
    freq_items = set(header_table.keys())
    if len(freq_items) == 0:
        return None, None
    
    for item in header_table:
        header_table[item] = [header_table[item], None]
    
    root = TreeNode('Null', 1, None)
    for transaction, count in dataset.items():
        sorted_items = [item for item in transaction if item in freq_items]
        sorted_items.sort(key=lambda x: header_table[x][0], reverse=True)
        if len(sorted_items) > 0:
            update_tree(sorted_items, root, header_table, count)
    
    return root, header_table

def update_tree(items, node, header_table, count):
    if items[0] in node.children:
        node.children[items[0]].increment_count(count)
    else:
        new_node = TreeNode(items[0], count, node)
        node.children[items[0]] = new_node
        if header_table[items[0]][1] is None:
            header_table[items[0]][1] = new_node
        else:
            update_header(header_table[items[0]][1], new_node)
    
    if len(items) > 1:
        update_tree(items[1:], node.children[items[0]], header_table, count)

def update_header(node_to_test, target_node):
    while node_to_test.next is not None:
        node_to_test = node_to_test.next
    node_to_test.next = target_node

def ascend_tree(node, prefix_path):
    if node.parent is not None:
        prefix_path.append(node.item)
        ascend_tree(node.parent, prefix_path)

def find_prefix_path(base_item, tree_node):
    conditional_patterns = {}
    while tree_node is not None:
        prefix_path = []
        ascend_tree(tree_node, prefix_path)
        if len(prefix_path) > 1:
            conditional_patterns[frozenset(prefix_path[1:])] = tree_node.count
        tree_node = tree_node.next
    return conditional_patterns

def mine_fptree(header_table, min_support, prefix, frequent_patterns):
    sorted_items = [item[0] for item in sorted(header_table.items(), key=lambda x: x[1][0])]
    for item in sorted_items:
        new_freq_set = prefix.copy()
        new_freq_set.add(item)
        frequent_patterns.append(new_freq_set)
        
        conditional_pattern_bases = find_prefix_path(item, header_table[item][1])
        conditional_tree, conditional_header = create_fptree(conditional_pattern_bases, min_support)
        
        if conditional_header is not None:
            mine_fptree(conditional_header, min_support, new_freq_set, frequent_patterns)

def fpgrowth(dataset, min_support):
    fp_tree, header_table = create_fptree(dataset, min_support)
    frequent_patterns = []
    mine_fptree(header_table, min_support, set(), frequent_patterns)
    return frequent_patterns

if __name__ == '__main__':
    # 測試資料集
    dataset = {
        frozenset(['a', 'b', 'c', 'd']): 2,
        frozenset(['b', 'c', 'e']): 3,
        frozenset(['a', 'b', 'c', 'e']): 2,
        frozenset(['a', 'b']): 2,
        frozenset(['a', 'c', 'd']): 2,
    }
    
    min_support = 2
    freq_patterns = fpgrowth(dataset, min_support)
    
    print("Frequent Patterns:")
    for pattern in freq_patterns:
        support = sum([dataset.get(trans, 0) for trans in dataset.keys() if pattern.issubset(trans)])
        print(pattern, ":", support)

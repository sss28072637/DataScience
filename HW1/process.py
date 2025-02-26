# 定义一个空字典用于存储每个项目数量的项目集
itemsets = {}

# 读取输出文件
output_file = "output_temp_myself.txt"  # 替换为你的输出文件路径

with open(output_file, 'r') as file:
    lines = file.readlines()

# 遍历每一行，按照项目数量分类
for line in lines:
    items = line.strip().split(':')  # 用冒号分割
    itemset = items[0].split(',')  # 用逗号分割项目集
    support = items[1]  # 支持度

    # 获取项目数量
    num_items = len(itemset)

    # 将项目集添加到对应数量的字典中
    if num_items not in itemsets:
        itemsets[num_items] = []
    itemsets[num_items].append((itemset, support))

# 按照项目数量的顺序输出
for num_items in sorted(itemsets.keys()):
    # print(f"{num_items}-itemset:")
    for itemset, support in itemsets[num_items]:
        print(f"{','.join(itemset)}:{support}")

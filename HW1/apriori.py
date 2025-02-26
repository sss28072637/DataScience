from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import pandas as pd

# 读取 input.txt 文件中的数据，每个 item 用逗号分隔
dataset = []
with open('input.txt', 'r') as file:
    for line in file:
        # 分割每行数据，移除空白符，然后转换为整数
        transaction = list(map(int, line.strip().split(',')))
        dataset.append(transaction)

# 使用 TransactionEncoder 来将数据转换成适合 Apriori 输入的形式
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 使用 Apriori 找出频繁项集
frequent_itemsets = apriori(df, min_support=0.15, use_colnames=True)

# 将结果按照 itemset 排序
frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: sorted(list(x)))

# 遍历输出结果
for index, row in frequent_itemsets.iterrows():
    itemset = ",".join([str(i) for i in row['itemsets']])
    support = "{:.4f}".format(row['support'])  # 格式化支持度为小数点后四位
    print(itemset + ":" + support)

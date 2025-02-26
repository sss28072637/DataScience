import sys
from collections import defaultdict

class Node:
    def __init__(self, item, count, parent):
        self.item = item;
        self.count = count;
        self.parent = parent;
        self.next = None;

def create_header_table(transactions, min_support):
    header_table = defaultdict(float)
    total_transactions = len(transactions)

    for transaction in transactions:
        for item in transaction:
            header_table[item] += 1
    
    header_table = {k: v/total_transactions for k, v in header_table.items() if v/total_transactions >= min_support}
    header_table = {item: [sup, None] for item, sup in header_table.items()}

    return header_table


def main():

    min_support = float(sys.argv[1])
    input_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    transactions = []
    with open(input_file_name, 'r') as file:
        for line in file:
            transaction = [int(item) for item in line.strip().split(',')]
            transactions.append(transaction)


    header_table = create_header_table(transactions, min_support)
    fptree = create_fp_tree()




if __name__ == "__main__":
    main()

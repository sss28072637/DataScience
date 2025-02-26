#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <algorithm>

struct Node {
    std::string item;
    int count;
    Node* parent;
    std::vector<Node*> children;

    Node(const std::string& val, int c, Node* p) : item(val), count(c), parent(p) {}
};

class FPGrowth {
private:
    double minSupp; // 最小支持度閾值
    std::unordered_map<std::string, int> singleItems;
    std::vector<std::vector<std::string>> transactions;
    Node* rootNode;

    void readTransactionsFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Unable to open file " << filename << std::endl;
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            std::vector<std::string> transaction;
            std::istringstream iss(line);
            std::string item;
            while (std::getline(iss, item, ',')) {
                transaction.push_back(item);
                singleItems[item]++;
            }
            transactions.push_back(transaction);
        }
        file.close();
    }

    void buildFPTree() {
        rootNode = new Node("null", 0, nullptr);
        for (const auto& transaction : transactions) {
            Node* curr = rootNode;
            for (const auto& item : transaction) {
                Node* child = nullptr;
                for (auto& ch : curr->children) {
                    if (ch->item == item) {
                        child = ch;
                        break;
                    }
                }
                if (child == nullptr) {
                    child = new Node(item, 1, curr);
                    curr->children.push_back(child);
                } else {
                    child->count++;
                }
                curr = child;
            }
        }
    }

    void mineTree(Node* node, const std::vector<std::string>& prefix, std::vector<std::vector<std::string>>& result) {
        std::unordered_map<std::string, int> items;
        while (node != nullptr) {
            if (node->item != "null") {
                std::vector<std::string> freqSet = prefix;
                freqSet.push_back(node->item);
                result.push_back(freqSet);
                items[node->item] = node->count;
            }
            node = node->parent;
        }

        std::vector<std::pair<std::string, int>> sortedItems(items.begin(), items.end());
        std::sort(sortedItems.begin(), sortedItems.end(), [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b) {
            return a.second > b.second;
        });

        for (const auto& pair : sortedItems) {
            std::vector<std::string> newPrefix = prefix;
            newPrefix.push_back(pair.first);
            std::vector<std::vector<std::string>> newTransactions;
            Node* newNode = nullptr;
            for (const auto& transaction : transactions) {
                std::vector<std::string> newTransaction;
                for (const auto& item : transaction) {
                    if (item == pair.first) {
                        newTransaction.push_back(item);
                    }
                }
                if (!newTransaction.empty()) {
                    newTransactions.push_back(newTransaction);
                }
            }
            if (newTransactions.size() > 0) {
                newNode = new Node(pair.first, pair.second, node);
                mineTree(newNode, newPrefix, result);
            }
        }
    }

public:
    FPGrowth(double minSupport) : minSupp(minSupport), rootNode(nullptr) {}

    void runFPGrowth(const std::string& filename) {
        readTransactionsFromFile(filename);
        buildFPTree();
        std::vector<std::vector<std::string>> result;
        mineTree(rootNode, {}, result);
        for (const auto& set : result) {
            for (const auto& item : set) {
                std::cout << item << " ";
            }
            std::cout << std::endl;
        }
    }

    void printTransactions() {
        std::cout << "Transactions:" << std::endl;
        for (const auto& transaction : transactions) {
            for (const auto& item : transaction) {
                std::cout << item << " ";
            }
            std::cout << std::endl;
        }
    }
};

int main() {
    FPGrowth fp(2); // 最小支持度閾值為2
    fp.runFPGrowth("transactions.txt"); // 檔案名稱包含交易資料
    fp.printTransactions(); // 印出交易
    return 0;
}

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <sstream>
#include <algorithm>

using namespace std;

vector<int> splitStringToInt(const string& s, char delimiter) {
    vector<int> tokens;
    stringstream ss(s);
    string token;
    while (getline(ss, token, delimiter)) {
        tokens.push_back(stoi(token));
    }
    return tokens;
}

string setToString(const set<int>& itemset) {
    stringstream ss;
    for (int item : itemset) {
        ss << item << " ";
    }
    return ss.str();
}

vector<vector<int>> readTransactions(const string& filename) {
    vector<vector<int>> transactions;
    ifstream file(filename);
    if (file.is_open()) {
        string line;
        while (getline(file, line)) {
            vector<int> transaction = splitStringToInt(line, ',');
            transactions.push_back(transaction);
        }
        file.close();
    }
    return transactions;
}

int countSupport(const vector<vector<int>>& transactions, const set<int>& candidate) {
    int count = 0;
    for (const vector<int>& transaction : transactions) {
        if (includes(transaction.begin(), transaction.end(), candidate.begin(), candidate.end())) {
            count++;
        }
    }
    return count;
}

void apriori(const vector<vector<int>>& transactions, float minSupport) {
    map<set<int>, float> frequentItemsets;  // 頻繁項目集合及其支持度
    map<set<int>, float> candidateItemsets; // 候選項目集合及其支持度

    // 初始化第一個候選項目集合
    set<int> candidate;
    for (const vector<int>& transaction : transactions) {
        for (int item : transaction) {
            set<int> temp = {item};
            candidateItemsets[temp] = 0;
        }
    }

    // 計算候選項目集合的支持度
    for (const auto& [candidate, count] : candidateItemsets) {
        int support = countSupport(transactions, candidate);
        candidateItemsets[candidate] = support;
        if (support >= minSupport) {
            frequentItemsets[candidate] = support;
        }
    }

    // Apriori主要迴圈
    while (!candidateItemsets.empty()) {
        map<set<int>, float> newCandidates;
        for (auto it1 = candidateItemsets.begin(); it1 != candidateItemsets.end(); ++it1) {
            for (auto it2 = next(it1); it2 != candidateItemsets.end(); ++it2) {
                set<int> newCandidate = it1->first;
                newCandidate.insert(it2->first.begin(), it2->first.end());
                if (newCandidate.size() == it1->first.size() + 1) {
                    int support = countSupport(transactions, newCandidate);
                    if (support >= minSupport) {
                        newCandidates[newCandidate] = support;
                        frequentItemsets[newCandidate] = support;
                    }
                }
            }
        }
        candidateItemsets = newCandidates;
    }

    // 輸出頻繁項目集合及其支持度
    // cout << "Frequent Itemsets:" << endl;
    for (const auto& [itemset, support] : frequentItemsets) {
        float freq = static_cast<float>(support) / transactions.size();
        if(freq >= minSupport)
            cout << setToString(itemset) << " : " << std::fixed << std::setprecision(4) << freq << endl;
        else
            continue;
    }
}

int main() {
    // 讀取交易記錄
    vector<vector<int>> transactions = readTransactions("sample.txt");

    // 設定最小支持度
    float minSupport = 0.2;

    // 執行Apriori演算法
    apriori(transactions, minSupport);

    return 0;
}

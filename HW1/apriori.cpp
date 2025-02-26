#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <set>
#include <sstream>
#include <algorithm>

// Function to generate candidates of size k from the previous frequent itemsets
std::vector<std::set<int>> generateCandidates(std::vector<std::set<int>>& prevCandidates, int k) {
    std::vector<std::set<int>> candidates;
    for (size_t i = 0; i < prevCandidates.size(); ++i) {
        for (size_t j = i + 1; j < prevCandidates.size(); ++j) {
            std::set<int> candidate = prevCandidates[i];
            candidate.insert(prevCandidates[j].begin(), prevCandidates[j].end());
            if (candidate.size() == k) {
                candidates.push_back(candidate);
            }
        }
    }
    return candidates;
}

// Function to calculate support of an itemset in transactions
int getSupport(std::vector<std::set<int>>& transactions, std::set<int>& itemset) {
    int count = 0;
    for (auto& transaction : transactions) {
        if (std::includes(transaction.begin(), transaction.end(), itemset.begin(), itemset.end())) {
            count++;
        }
    }
    return count;
}

// Function to perform Apriori algorithm
std::vector<std::set<int>> apriori(std::vector<std::set<int>>& transactions, double minSupport) {
    // Step 1: Generate 1-itemsets
    std::unordered_map<int, int> itemSupports;
    for (auto& transaction : transactions) {
        for (auto& item : transaction) {
            itemSupports[item]++;
        }
    }

    // Step 2: Find frequent 1-itemsets
    std::vector<std::set<int>> frequentItemsets;
    for (auto& pair : itemSupports) {
        double support = static_cast<double>(pair.second) / transactions.size();
        if (support >= minSupport) {
            frequentItemsets.push_back({pair.first});
        }
    }

    // Step 3: Generate frequent itemsets of size > 1
    int k = 2;
    std::vector<std::set<int>> prevFrequentItemsets = frequentItemsets;
    while (!prevFrequentItemsets.empty()) {
        std::vector<std::set<int>> candidates = generateCandidates(prevFrequentItemsets, k);
        prevFrequentItemsets.clear();

        for (auto& candidate : candidates) {
            int support = getSupport(transactions, candidate);
            double supportRatio = static_cast<double>(support) / transactions.size();
            if (supportRatio >= minSupport) {
                frequentItemsets.push_back(candidate);
                prevFrequentItemsets.push_back(candidate);
            }
        }
        k++;
    }

    return frequentItemsets;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <input_filename> <output_filename> <min_support>" << std::endl;
        return 1;
    }

    double minSupport = std::stod(argv[1]);
    std::string inputFilename = argv[2];
    std::string outputFilename = argv[3];
    

    std::ifstream inputFile(inputFilename);
    if (!inputFile.is_open()) {
        std::cerr << "Error opening file: " << inputFilename << std::endl;
        return 1;
    }

    std::vector<std::set<int>> transactions;
    std::string line;
    while (std::getline(inputFile, line)) {
        std::set<int> transaction;
        std::istringstream iss(line);
        int item;
        while (iss >> item) {
            transaction.insert(item);
        }
        transactions.push_back(transaction);
    }
    inputFile.close();

    // Perform Apriori
    std::vector<std::set<int>> frequentItemsets = apriori(transactions, minSupport);

    // Write results to output file
    std::ofstream outputFile(outputFilename);
    if (!outputFile.is_open()) {
        std::cerr << "Error opening file for writing: " << outputFilename << std::endl;
        return 1;
    }

    for (auto& itemset : frequentItemsets) {
        for (auto& item : itemset) {
            outputFile << item << " ";
        }
        outputFile << std::endl;
    }

    outputFile.close();

    std::cout << "Frequent itemsets written to " << outputFilename << std::endl;

    return 0;
}

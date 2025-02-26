#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <set>
#include <map>
#include <iterator>
#include <algorithm>
#include <iomanip>
// #include <chrono> // for time recording

using namespace std;

// int total_items = 0;

float calculate_frquency(vector<vector<int>> transactions, set<int> itemset) {
    int cnt = 0;
    for(vector<int> cur_transaction: transactions) {
        bool found = true;

        for(int item: itemset) {
            if(find(cur_transaction.begin(), cur_transaction.end(), item) == cur_transaction.end()) {
                found = false;
                break;
            }
        }   

        if(found) {
            cnt += 1;
        }        
    }

    float frequency = static_cast<float>(cnt) / transactions.size();

    return frequency;
}


map<set<int>, float> find_frequent_itemsets(vector<vector<int>> transactions, float min_support) {
    map<set<int>, float> frequent_itemsets;
    map<set<int>, float> candidate_itemsets;

    for(vector<int> cur_transaction: transactions) {
        for(int item: cur_transaction) {
            candidate_itemsets[{item}] = 0;
        }
    }

    for(map<set<int>, float>::iterator it=candidate_itemsets.begin(); it != candidate_itemsets.end(); ++it) {
        set<int> itemset = it->first;
        float frequency = calculate_frquency(transactions, itemset);

        // cout << frequency << endl;

        candidate_itemsets[itemset] = frequency;
        if(frequency >= min_support) {
            // cout << frequency << endl;
            frequent_itemsets[itemset] = frequency;
        }
    }

    map<set<int>, float> new_candidates;
    // for(; candidate_itemsets.empty() == false; candidate_itemsets = new_candidates) {
    // while(!candidate_itemsets.empty()) {
    while(1) {

        if(candidate_itemsets.empty()) {
            break;
        }

        for(auto outer_it = candidate_itemsets.begin(); outer_it != candidate_itemsets.end(); ++outer_it) {
            set<int> outer_set = outer_it -> first;
            for(auto inner_it = next(outer_it); inner_it != candidate_itemsets.end(); ++inner_it) {
                set<int> inner_set = inner_it -> first;
                
                set<int> new_itemset;
                set_union(outer_set.begin(), outer_set.end(), inner_set.begin(), inner_set.end(), inserter(new_itemset, new_itemset.begin()));
            
                if(outer_set.size() + 1 == new_itemset.size()) {
                    float new_frequency = calculate_frquency(transactions, new_itemset);
                    if(new_frequency >= min_support) {
                        new_candidates[new_itemset] = new_frequency;
                        frequent_itemsets[new_itemset] = new_frequency;
                    }
                }
            }
        }
        candidate_itemsets = new_candidates;
        new_candidates.clear();

    }

    return frequent_itemsets;
}

string print_set(set<int> itemset) {
    stringstream ss;
    for(auto it = itemset.begin(); it != itemset.end(); ++it) {
        if(it != itemset.begin()) {
            ss << ",";
        }
        ss << *it;
    }
    return ss.str();
}

void write_file(string filename, map<set<int>, float> freq_items) {
    ofstream outfile(filename);

    for(auto [itemset, freq]: freq_items) {
        outfile << print_set(itemset) << ":" << fixed << setprecision(4) << freq << endl;
    }

    outfile.close();
}

int main(int argc, char *argv[]) {

    // auto start = chrono::high_resolution_clock::now();
    
    double min_support = stod(argv[1]);
    string input_filename = argv[2];
    string output_filename = argv[3];

    ifstream input_file(input_filename);

    vector<vector<int>> transactions_data;

    // Read Transactions data
    while(1) {
        string line;

        if(!getline(input_file, line)) {
            break;
        }

        vector<int> cur_transaction;
        stringstream ss(line);
        string item;

        while(getline(ss, item, ',')) {
            int item_int = stoi(item);
            cur_transaction.push_back(item_int);
            // total_items += 1;
        }

        transactions_data.push_back(cur_transaction);
    }

    input_file.close();


    map<set<int>, float> frequent_items;
    frequent_items = find_frequent_itemsets(transactions_data, min_support);

    write_file(output_filename, frequent_items);

    // auto end = chrono::high_resolution_clock::now();

    // chrono::duration<double> duration = end - start;

    // cout << "Execution Time: " << duration.count() << " seconds" << endl;

    return 0;
}
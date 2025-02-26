rm 109000205_hw1
g++ -std=c++2a -pthread -O2 -o 109000205_hw1 109000205_hw1.cpp
./109000205_hw1 0.15 input.txt output_temp_myself.txt
python process.py > output_myself.txt
python apriori.py > output_library.txt
diff output_myself.txt output_library.txt

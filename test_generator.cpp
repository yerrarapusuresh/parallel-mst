#include <iostream>
#include <random>
#include <math.h>

const int MAX = 20000;

int graph[MAX][MAX];

int main(){
		int VERTICES_NUMBER,c;

		// graph density depends on the value of the constant c
		std::cin >> VERTICES_NUMBER >> c;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<> real_dis(0, 1);
    std::uniform_int_distribution<> int_dis(1,200);

    double a = c*log(VERTICES_NUMBER)/VERTICES_NUMBER;

    int m = 0;

    for(int i=0; i<VERTICES_NUMBER; ++i)
        for(int j=i+1; j<VERTICES_NUMBER; ++j){
            double v = real_dis(gen);
            if(v<a){
                m += 2;
                int weight = int_dis(gen);
                graph[i][j] = weight;
                graph[j][i] = weight;
            }
        }
    
    std::cout << VERTICES_NUMBER << " " << m << std::endl;

    for(int i=0; i<VERTICES_NUMBER; ++i)
        for(int j=0; j<VERTICES_NUMBER; ++j)
            if(graph[i][j] != 0) std::cout << i << " ";

    std::cout << std::endl;

    for(int i=0; i<VERTICES_NUMBER; ++i)
        for(int j=0; j<VERTICES_NUMBER; ++j)
            if(graph[i][j] != 0) std::cout << j << " ";

    std::cout << std::endl;

    for(int i=0; i<VERTICES_NUMBER; ++i)
        for(int j=0; j<VERTICES_NUMBER; ++j)
            if(graph[i][j] != 0) std::cout << graph[i][j] << " ";

    std::cout << std::endl;

    return 0;
}


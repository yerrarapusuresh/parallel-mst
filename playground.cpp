#include <iostream>
#include <algorithm>
#include <vector>
#include <unistd.h>
#include <stdint.h>
#include "mst.h"

#define f first
#define s second

const int MAX = 1 << 25;
bool vis[MAX];
std::vector<uint32_t> G[MAX];

int n,m;
uint32_t *E1,*E2,*W,*flags;

int rep[MAX];
std::vector< std::pair<int,int> > graph[MAX];

int Find(int a){
    if(rep[a]==a) return a;
    int ra=Find(rep[a]);
    rep[a]=ra;
    return ra;
}

void Union(int a, int b){
    int ra=Find(a);
    int rb=Find(b);
    rep[ra]=rb;
}

int Kruskal(){
    int res = 0;
    std::vector< std::pair< uint32_t, std::pair<uint32_t,uint32_t> > > pom;

    for(int i=0; i<n; ++i) rep[i] = i;

    for(int i=0; i<m; ++i)
        pom.push_back(std::make_pair(W[i],std::make_pair(E1[i],E2[i])));

    std::sort(pom.begin(),pom.end());

    for(int i=0; i<m; ++i)
        if(Find(pom[i].s.f) != Find(pom[i].s.s)){
            res += pom[i].f;
            Union(pom[i].s.f,pom[i].s.s);
        }

    return res;
}

void read_graph(){
    std::cin >> n >> m;

    E1 = new uint32_t[m];
    E2 = new uint32_t[m];
    W = new uint32_t[m];
    flags = new uint32_t[m];

    for(int i=0; i<m; ++i) flags[i] = 0;

    for(int i=0; i<m; ++i) std::cin >> E1[i];
    for(int i=0; i<m; ++i) std::cin >> E2[i];
    for(int i=0; i<m; ++i) std::cin >> W[i];
}

void dfs(int x){
    vis[x] = true;

    for(int i=0; i<G[x].size(); ++i)
        if(!vis[G[x][i]]) dfs(G[x][i]);
}

bool is_connected(int n){
    for(int i=0; i<n; ++i) vis[i] = false;

    dfs(0);

    for(int i=0; i<n; ++i)
        if(!vis[i]) return false;

    return true;
}

bool checker(uint32_t *E1, uint32_t *E2, uint32_t *W, uint32_t *flags, int n, int m){
    int cost = 0;
    int counter = 0;

    for(int i=0; i<m; ++i)
        if(flags[i]==1){
            cost += W[i];
            G[E1[i]].push_back(E2[i]);
            G[E2[i]].push_back(E1[i]);
            ++counter;
        }

    counter /= 2;
    cost /= 2;

    clock_t start,finish;

    start = clock();
    int optimal_cost = Kruskal();
    finish = clock();
   
    if(counter > n-1) std::cout << "TOO MANY EDGES";
    else
        if(!is_connected(n)) std::cout << "NOT CONNECTED";
        else
            if(cost != optimal_cost) std::cout << "TREE IS NOT MINIMAL";
            else
                std::cout << "OK";

    std::cout << std::endl << "Sequential algorithm execution time: " << (double)(finish-start)/CLOCKS_PER_SEC << std::endl;
}

int main(){
    read_graph();
    clock_t start,finish;
    start = clock();
    uint32_t *tree = parallel_mst(E1,E2,W,n,m);
    finish = clock();
    checker(E1,E2,W,tree,n,m);

    std::cout << "Parallel algorithm execution time: " << (double)(finish-start)/CLOCKS_PER_SEC << std::endl;

    return 0;
}


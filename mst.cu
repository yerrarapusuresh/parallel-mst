#include <thrust/scan.h>
#include <thrust/sort.h>
#include <unistd.h>
#include <stdint.h>
#include "radixsort.cuh"
using namespace std;

const uint64_t MASK = (1<<24) - 1;
const uint64_t MASK2 = (1ULL<<32) - 1;

uint32_t op(uint32_t x, uint32_t y){
    if(x<y) return x;
    else
        return y;
}

bool pred(uint32_t x, uint32_t y){
    if(x==y) return true;
    return false;
}

__global__
void fill(uint32_t *I, int m){
    int thid = threadIdx.x + 1024*blockIdx.x;

    if(thid<m){
        I[thid] = thid;
    }
}

__global__
void mark_taken_edges(uint32_t *res, uint32_t *S, uint32_t *V, uint32_t *E, uint32_t *W, int m){
    int thid = threadIdx.x + 1024*blockIdx.x;

    if(thid<m){
        res[thid] = 0;
        uint32_t a = V[thid];
        uint32_t b = E[thid];
        if(S[a]==b||S[b]==a) res[thid] = 1;
    }
}

__global__
void mark_edges_from_rec(uint32_t *old_labels, uint32_t *rec_output, uint32_t *output, int m){
    int thid = threadIdx.x + 1024*blockIdx.x;
    
    if(thid<m){
        if(rec_output[thid] == 1){
            output[old_labels[thid]] = 1;
        }
    }
}

__global__
void transform(uint32_t *X, uint32_t *E, uint32_t *W, int m){
    int thid = threadIdx.x + 1024*blockIdx.x;
    if(thid<m) X[thid] = (W[thid] << 24) + E[thid];
}

__global__
void find_successor(uint32_t *S, uint32_t *NWE, int n){
    int thid = threadIdx.x + 1024*blockIdx.x;
    if(thid<n) S[thid] = NWE[thid] & MASK;
}

__global__
void remove_cycle(uint32_t *S, int n){
    int thid = threadIdx.x + 1024*blockIdx.x;

    if(thid<n){
        uint32_t a = S[thid];
        uint32_t b = S[S[thid]];
        if(thid==b && thid<a) S[thid] = thid;
    }
}

__global__
void merge_vertices(uint32_t *S, int n){
    int thid = threadIdx.x + 1024*blockIdx.x;

    if(thid<n){
        while(S[thid] != S[S[thid]]){
            S[thid] = S[S[thid]];
        }
    }
}

__global__
void form_list(uint64_t *L, uint32_t *S, int n){
    int thid = threadIdx.x + 1024*blockIdx.x;

    if(thid<n){
        L[thid] = (S[thid]);
        L[thid] <<= 32;
        L[thid] += thid;
    }
}

__global__
void connect(uint32_t *C, uint64_t *L, int n){
    int thid = threadIdx.x + 1024*blockIdx.x;

    if(thid<n){
        if(thid==0){
            C[0] = 1;
            return ;
        }
        uint64_t a = L[thid-1] >> 32;
        uint64_t b = L[thid] >> 32;
        if(a != b) C[thid] = 1;
        else
            C[thid] = 0;
    }
}

__global__
void relabel(uint32_t *new_v, uint64_t *L, uint32_t *C, int n){
    int thid = threadIdx.x + 1024*blockIdx.x;

    if(thid<n) new_v[L[thid] & MASK2] = C[thid] - 1;
}

__global__
void make_triples(uint64_t *triples, uint32_t *E, uint32_t *W, uint32_t *new_v, uint32_t *F, int m){
    int thid = threadIdx.x + 1024*blockIdx.x;

    if(thid<m)
        triples[thid] = ((uint64_t)new_v[F[thid]] << 40) + ((uint64_t)new_v[E[thid]] << 16) + W[thid];
}

const uint64_t MASK3 = ULONG_MAX - ((1<<16) - 1);
const uint64_t MASK4 = ((1<<24) - 1);
const uint64_t MASK5 = (1<<16) - 1;

__device__
bool non_degenerate(uint64_t x){
    uint64_t a = x >> 40;
    uint64_t b = x >> 16;
    b &= MASK4;
    if(a != b) return true;
    return false;
}

__global__
void mark_first(uint32_t *idx, uint64_t *triples, int m){
    int thid = threadIdx.x + 1024*blockIdx.x;
   
    if(thid<m){
        idx[thid] = 0;
        if(non_degenerate(triples[thid]))
            if(thid==0 || (triples[thid]&MASK3) != (triples[thid-1]&MASK3)){
                idx[thid] = 1;
            }
    }
}

__global__
void reduce_WE(uint32_t *reduced_W, uint32_t *reduced_E, uint32_t *tmp_V, uint32_t *flags, uint32_t *idx, uint64_t *triples, uint32_t *labels, uint32_t *old_labels, int m){
    int thid = threadIdx.x + 1024*blockIdx.x;

    if(thid<m){
        if(flags[thid]==1){
            reduced_E[idx[thid] - 1] = (triples[thid] >> 16) & MASK4;
            reduced_W[idx[thid] - 1] = triples[thid] & MASK5;
            old_labels[idx[thid] - 1] = labels[thid];
            tmp_V[idx[thid] - 1] = (triples[thid] >> 40);
        }
    }
}

__global__
void set_mins(uint32_t *NWE, uint32_t *X, uint32_t *F, int m){
    int thid = threadIdx.x + 1024*blockIdx.x;
    if(thid<m)
        if(thid==m-1 || F[thid] != F[thid+1]) NWE[F[thid]] = X[thid];
}

void find_min(uint32_t *NWE, uint32_t *X, uint32_t *F, uint32_t *device_F, int m){
    thrust::inclusive_scan_by_key(F,F+m,X,X,pred,op);

    uint32_t *copy_X;
    cudaMalloc((void**)&copy_X,sizeof(uint32_t)*m);
    cudaMemcpy(copy_X,X,sizeof(uint32_t)*m,cudaMemcpyHostToDevice);
    cudaThreadSynchronize();

    set_mins<<<(m+1023)/1024,1024>>>(NWE,copy_X,device_F,m);
    cudaThreadSynchronize();

    cudaFree(copy_X);
}

void mst(uint32_t *output, uint32_t *V, uint32_t *E, uint32_t *W, int n, int m){
    if(n<=1) return;

    int m_blocks_number = (m+1023)/1024;
    int n_blocks_number = (n+1023)/1024;

    uint32_t *X;
    cudaMalloc((void**)&X,sizeof(uint32_t)*m);
    cudaThreadSynchronize();

    transform<<<m_blocks_number,1024>>>(X,E,W,m);
    cudaThreadSynchronize();
    
    uint32_t *F;
    F = new uint32_t[m];
    cudaMemcpy(F,V,sizeof(uint32_t)*m,cudaMemcpyDeviceToHost);

    uint32_t *host_X;
    host_X = new uint32_t[m];
    cudaMemcpy(host_X,X,sizeof(uint32_t)*m,cudaMemcpyDeviceToHost);

    uint32_t *NWE;
    cudaMalloc((void**)&NWE,sizeof(uint32_t)*n);

    cudaThreadSynchronize();
    find_min(NWE,host_X,F,V,m);

    delete F;
    delete host_X;
    cudaFree(X);

    uint32_t *S;
    cudaMalloc((void**)&S,sizeof(uint32_t)*n);
    cudaThreadSynchronize();

    find_successor<<<n_blocks_number,1024>>>(S,NWE,n);
    cudaThreadSynchronize();

    cudaFree(NWE);

    remove_cycle<<<n_blocks_number,1024>>>(S,n);
    cudaThreadSynchronize();

    mark_taken_edges<<<m_blocks_number,1024>>>(output,S,V,E,W,m);
    cudaThreadSynchronize();

    merge_vertices<<<n_blocks_number,1024>>>(S,n);
    cudaThreadSynchronize();

    uint64_t *L,*l;
    cudaMalloc((void**)&L,sizeof(uint64_t)*n);
    cudaThreadSynchronize();

    l = new uint64_t[n];

    form_list<<<n_blocks_number,1024>>>(L,S,n);
    cudaThreadSynchronize();

    cudaFree(S);

    cudaMemcpy(l,L,sizeof(uint64_t)*n,cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    thrust::sort(l,l+n);

    cudaMemcpy(L,l,sizeof(uint64_t)*n,cudaMemcpyHostToDevice);

    delete l;

    uint32_t *C;
    cudaMalloc((void**)&C,sizeof(uint32_t)*n);
    cudaThreadSynchronize();

    connect<<<n_blocks_number,1024>>>(C,L,n);
    cudaThreadSynchronize();

    uint32_t *c;
    c = new uint32_t[n];
    cudaMemcpy(c,C,sizeof(uint32_t)*n,cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    thrust::inclusive_scan(c,c+n,c);

    cudaMemcpy(C,c,sizeof(uint32_t)*n,cudaMemcpyHostToDevice);

    delete c;

    uint32_t *new_v;
    cudaMalloc((void**)&new_v,sizeof(uint32_t)*n);
    cudaThreadSynchronize();

    relabel<<<n_blocks_number,1024>>>(new_v,L,C,n);
    cudaThreadSynchronize();

    cudaFree(C);
    cudaFree(L);

    uint64_t *triples;
    cudaMalloc((void**)&triples,sizeof(uint64_t)*m);
    cudaThreadSynchronize();

    make_triples<<<m_blocks_number,1024>>>(triples,E,W,new_v,V,m);
    cudaThreadSynchronize();

    cudaFree(new_v);

    uint32_t *I;
    cudaMalloc((void**)&I,sizeof(uint32_t)*m);
    cudaThreadSynchronize();

    fill<<<m_blocks_number,1024>>>(I,m);
    cudaThreadSynchronize();

    radixsort(triples,triples,I,m);
    cudaThreadSynchronize();

    uint32_t *idx;
    cudaMalloc((void**)&idx,sizeof(uint32_t)*m);

    uint32_t *flags;
    cudaMalloc((void**)&flags,sizeof(uint32_t)*m);
    cudaThreadSynchronize();

    mark_first<<<m_blocks_number,1024>>>(flags,triples,m);
    cudaThreadSynchronize();

    uint32_t *Flags;
    Flags = new uint32_t[m];

    cudaMemcpy(Flags,flags,sizeof(uint32_t)*m,cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
   
    thrust::inclusive_scan(Flags,Flags+m,Flags);

    cudaMemcpy(idx,Flags,sizeof(uint32_t)*m,cudaMemcpyHostToDevice);
    cudaThreadSynchronize();

    int new_m = Flags[m-1];

    delete Flags;

    uint32_t *old_labels;
    cudaMalloc((void**)&old_labels,sizeof(uint32_t)*new_m);

    uint32_t *reduced_E;
    cudaMalloc((void**)&reduced_E,sizeof(uint32_t)*new_m);

    uint32_t *reduced_W;
    cudaMalloc((void**)&reduced_W,sizeof(uint32_t)*new_m);

    uint32_t *reduced_V;
    cudaMalloc((void**)&reduced_V,sizeof(uint32_t)*new_m);
    cudaThreadSynchronize();

    reduce_WE<<<m_blocks_number,1024>>>(reduced_W,reduced_E,reduced_V,flags,idx,triples,I,old_labels,m);
    cudaThreadSynchronize();

    uint32_t *Reduced_V;
    Reduced_V = new uint32_t[new_m];

    cudaMemcpy(Reduced_V,reduced_V,sizeof(uint32_t)*new_m,cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    cudaFree(I);
    cudaFree(triples);
    cudaFree(idx);
    cudaFree(flags);

    uint32_t *new_output;
    cudaMalloc((void**)&new_output,sizeof(uint32_t)*m);
    cudaThreadSynchronize();

    mst(new_output,reduced_V,reduced_E,reduced_W,Reduced_V[new_m-1] + 1,new_m);

    delete Reduced_V;

    cudaFree(reduced_V);
    cudaFree(reduced_E);
    cudaFree(reduced_W);

    mark_edges_from_rec<<<(new_m+1023)/1024,1024>>>(old_labels,new_output,output,new_m);
    cudaThreadSynchronize();

    cudaFree(old_labels);
    cudaFree(new_output);
}

uint32_t* parallel_mst(uint32_t *E1, uint32_t *E2, uint32_t *W, int n, int m){
    uint32_t *e1,*e2,*w,*res;

    cudaMalloc((void**)&e1,sizeof(uint32_t)*m);
    cudaMalloc((void**)&e2,sizeof(uint32_t)*m);
    cudaMalloc((void**)&w,sizeof(uint32_t)*m);
    cudaMalloc((void**)&res,sizeof(uint32_t)*m);

    cudaMemcpy(e1,E1,sizeof(uint32_t)*m,cudaMemcpyHostToDevice);
    cudaMemcpy(e2,E2,sizeof(uint32_t)*m,cudaMemcpyHostToDevice);
    cudaMemcpy(w,W,sizeof(uint32_t)*m,cudaMemcpyHostToDevice);

    mst(res,e1,e2,w,n,m);

    uint32_t *Res = new uint32_t[m];
    cudaMemcpy(Res,res,sizeof(uint32_t)*m,cudaMemcpyDeviceToHost);

    cudaFree(e1);
    cudaFree(e2);
    cudaFree(w);
    cudaFree(res);

    return Res;
}


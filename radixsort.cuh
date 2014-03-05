#ifndef H_RADIXSORT
#define RADIXSORT

__global__
void bits_prefix_sum(uint64_t *A, uint64_t *B, uint64_t *T, int n, int pos, uint64_t bit){
    __shared__ int sum[1024];
    int thid = 1024*blockIdx.x+threadIdx.x;

    if(2*thid<n && (T[2*thid]&(1ULL<<pos))==bit) sum[threadIdx.x] = 1;
    else
        sum[threadIdx.x] = 0;

    if(2*thid+1<n && (T[2*thid+1]&(1ULL<<pos))==bit) ++sum[threadIdx.x];
    __syncthreads();

    for(int i=1; i<1024; i*=2){
        int v = sum[threadIdx.x];
        if(threadIdx.x>=i) v += sum[threadIdx.x-i];
        __syncthreads();
        sum[threadIdx.x] = v;
    }

    A[thid] = sum[threadIdx.x];
    if((threadIdx.x == 1023 && 2*thid+1<n) || 2*thid==n-1 || 2*thid==n-2) B[blockIdx.x] = sum[threadIdx.x];
}

__global__
void bits_update(uint64_t *A, uint64_t *B, int n){
    int thid = 1024*blockIdx.x+threadIdx.x;
    if(blockIdx.x>0 && thid<n) A[thid] += B[blockIdx.x-1];
}

__global__
void bits_order(uint64_t *A, uint64_t *B, uint64_t *C, uint64_t *D, uint32_t *idx, uint32_t *new_idx, int n, int pos){
    int thid = 1024*blockIdx.x+threadIdx.x;
    int v;
    int a = 0;

    if(2*thid<n){
        if((A[2*thid]&(1ULL<<pos))==0){
            if(thid != 0) v = C[thid-1];
            else
                v = 0;
            a = 1;
            B[v] = A[2*thid];
            new_idx[v] = idx[2*thid];
        }
        else{
            if(thid != 0) v = D[thid-1];
            else
                v = 0;
            B[C[(n+1)/2-1]+v] = A[2*thid];
            new_idx[C[(n+1)/2-1]+v] = idx[2*thid];

        }

        if(2*thid+1<n){
            if((A[2*thid+1]&(1ULL<<pos))==0){
                if(thid != 0) v = C[thid-1]+a;
                else
                    v = a;
                B[v] = A[2*thid+1];
                new_idx[v] = idx[2*thid+1];
            }
            else{
                if(thid != 0) v = D[thid-1];
                else
                    v = 0;
                B[C[(n+1)/2-1]+v+(a^1)] = A[2*thid+1];
                new_idx[C[(n+1)/2-1]+v+(a^1)] = idx[2*thid+1];
            }
        }
    }
}

void do_bits_prefix_sum(uint64_t* A, uint64_t* C, int n, int pos, uint64_t bit){
    int blocks_number = (n+2047)/2048;
    uint64_t *B;

    cudaMalloc((void**)&B,sizeof(uint64_t)*n);
    cudaThreadSynchronize();

    bits_prefix_sum<<<blocks_number,1024>>>(A,B,C,n,pos,bit);
    cudaThreadSynchronize();

    if(blocks_number > 1){
        uint64_t *t = new uint64_t[blocks_number];
        cudaMemcpy(t,B,sizeof(uint64_t)*blocks_number,cudaMemcpyDeviceToHost);

        int sum=0;
        for(int i=0; i<blocks_number; ++i){
            sum += t[i];
            t[i] = sum;
        }

        cudaMemcpy(B,t,sizeof(uint64_t)*blocks_number,cudaMemcpyHostToDevice);

        delete t;
    }

    bits_update<<<blocks_number,1024>>>(A,B,n);
    cudaThreadSynchronize();

    cudaFree(B);
}

void radixsort(uint64_t* dest, uint64_t* S, uint32_t *idx, int n){
    uint64_t *A,*B,*C,*D;
    uint32_t *new_idx;

    cudaMalloc((void**)&A,sizeof(uint64_t)*n);
    cudaMalloc((void**)&B,sizeof(uint64_t)*n);
    cudaMalloc((void**)&C,sizeof(uint64_t)*n);
    cudaMalloc((void**)&D,sizeof(uint64_t)*n);
    cudaMalloc((void**)&new_idx,sizeof(uint32_t)*n);

    cudaMemcpy(A,S,sizeof(uint64_t)*n,cudaMemcpyDeviceToDevice);

    int pos;
    int bits = 64;

    for(int i=0; i<bits; ++i){
        do_bits_prefix_sum(C,A,n,i,0);
        do_bits_prefix_sum(D,A,n,i,1ULL<<i);
        pos = i;
       
        bits_order<<<(n+1023)/1024,1024>>>(A,B,C,D,idx,new_idx,n,pos);
        cudaThreadSynchronize();

        cudaMemcpy(A,B,sizeof(uint64_t)*n,cudaMemcpyDeviceToDevice);
        cudaMemcpy(idx,new_idx,sizeof(uint32_t)*n,cudaMemcpyDeviceToDevice);

        cudaThreadSynchronize();
    }

    cudaMemcpy(dest,A,sizeof(uint64_t)*n,cudaMemcpyDeviceToDevice);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(D);
    cudaFree(new_idx);
}

#endif


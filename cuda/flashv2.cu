#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

// require Br = Bc
__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    //batch size * num_heads * N * d 就是第一个batch + num_heads * N * d 第一个head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S 
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* Oi = &sram[tile_size * 3]; // more sram space  for O
    float* S = &sram[tile_size * 4];
    

    for (int i = 0; i < Tr; i++) {
        for (int x = 0; x < d; x++) {
            Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            Oi[(tx * d) + x] = 0;
        }
        __syncthreads();  // such that the inner loop can use the correct Qi

        float row_m_prev = -INFINITY;
        float row_l_prev = 0;

        float row_l_new = 0;// li 0 
        float row_m = -INFINITY; // mi
        float row_m_new = -INFINITY; // 

        for (int j = 0; j < Tc; j++)  {
            for (int x = 0; x < d; x++) {
                Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
                Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
            }

            // S = QK^T, row_m = rowmax(S)
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;
                //每次算一列的结果, 一个thread算一行的结果. 就是偏移tx 

                if (sum > row_m)
                    row_m = sum; //mij更新
            }

            row_m_new = max(row_m_prev, row_m); 

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m); // in place update S to Pij to save space
                row_l += S[(Bc * tx) + y];
            }
            // lij 改变了. 

            // don't need Compute new m and l
            // float row_m_new = max(row_m_prev, row_m); 
            // float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + ( 1 * row_l); // remove the second expf, why?
            row_l_new =  (__expf(row_m_prev - row_m_new) * row_l_prev)  + row_l;

            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                Oi[(tx * d) + x] = (1 / __expf(row_m_prev - row_m_new) ) * Oi[(tx * d) + x]  + pv ;
            }

            row_m_prev = row_m_new;
            row_l_prev = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Qi in inner loop
        
        for (int x = 0; x < d; x++) {
            Oi[(tx * d) + x] = (1 / row_l_new) *  Oi[(tx * d) + d] ; //Oi Tc, 
        }

        
        // Write O, Li to HBM, don't need m.
        for (int x = 0; x < d; x++) {
            O[qkv_offset + (tile_size * i) + (tx * d) + x] = Oi[(tx * d) + x];
        }

        //Only need to store the logsumexp to reduce memory overhead while maintain correctness for the BWD pass.
        l[lm_offset + (Br * i) + tx] = row_m_new + log(row_l_new); // li = mi + log(li)
        
        // m[lm_offset + (Br * i) + tx] = row_m_new;
        // l[lm_offset + (Br * i) + tx] = row_l_new; // don't need write back HBM
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // TODO: determine Bc, Br dynamically
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    int Bc = ceil((float) N / Bc);
    // const int Bc = 32; const int Br = 32;// 根据head来决定的? 

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);// N is sequence length

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY); //don't need Extra 𝒪 𝑁 to store 𝑚.
    torch::Device device(torch::kCUDA);
    l = l.to(device); m = m.to(device);

    // Calculate SRAM size needed per block
    const int sram_size = (4 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh);  // batch_size x num_heads 个block.
    dim3 block_dim(Bc);  // Bc threads per block

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
    return O;
}
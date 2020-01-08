#include <cublasLt.h>
#include "THC/THC.h"

struct CublasLtStructs {
public:
  CublasLtStructs() :
    op_desc_(NULL),
    preference_(NULL),
    heuristic_({0}),
    mat_a_(NULL),
    mat_b_(NULL),
    mat_c_(NULL) { }

  void CreateStructs(cublasOperation_t trans_a, cublasOperation_t trans_b, int m, int n, int k, int lda, int ldb, int ldc)
  {
    THCublasCheck(cublasLtMatmulDescCreate(&op_desc_, CUDA_R_32F));
    THCublasCheck(cublasLtMatmulPreferenceCreate(&preference_));

	// Workspace Size Maximum
    uint64_t workspace_size = 16*1024; // bytes
    THCublasCheck(cublasLtMatmulPreferenceSetAttribute(preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    // Limit SplitK usage to in-place kernels
    cublasLtReductionScheme_t red_scheme = static_cast<cublasLtReductionScheme_t>(CUBLASLT_REDUCTION_SCHEME_NONE | CUBLASLT_REDUCTION_SCHEME_INPLACE);
    THCublasCheck(cublasLtMatmulPreferenceSetAttribute(preference_, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &red_scheme, sizeof(red_scheme)));

    THCublasCheck(cublasLtMatrixLayoutCreate(&mat_a_, CUDA_R_16F,  trans_a == CUBLAS_OP_N ? m : k, trans_a == CUBLAS_OP_N ? k : m, lda));
    THCublasCheck(cublasLtMatrixLayoutCreate(&mat_b_, CUDA_R_16F,  trans_b == CUBLAS_OP_N ? k : n, trans_b == CUBLAS_OP_N ? n : k, ldb));
    THCublasCheck(cublasLtMatrixLayoutCreate(&mat_c_, CUDA_R_16F,  m, n, ldc));

    THCublasCheck(cublasLtMatmulDescSetAttribute(op_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    THCublasCheck(cublasLtMatmulDescSetAttribute(op_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
  }

  void DestroyStructs()
  {
    if (op_desc_)    cublasLtMatmulDescDestroy(op_desc_);
    if (preference_) cublasLtMatmulPreferenceDestroy(preference_);
    if (mat_a_)      cublasLtMatrixLayoutDestroy(mat_a_);
    if (mat_b_)      cublasLtMatrixLayoutDestroy(mat_b_);
    if (mat_c_)      cublasLtMatrixLayoutDestroy(mat_c_);
  }

  cublasLtMatmulDesc_t            op_desc_;          
  cublasLtMatmulPreference_t      preference_;
  cublasLtMatmulHeuristicResult_t heuristic_;
  cublasLtMatrixLayout_t          mat_a_;
  cublasLtMatrixLayout_t          mat_b_;
  cublasLtMatrixLayout_t          mat_c_;
};

struct SelfAttnGemmStructs {
public:  
  SelfAttnGemmStructs() :
    instances_fwd_(0),
    instances_bwd_(0),
    batches_fwd_(0),
    batches_bwd_(0),
    input_lin_fwd_(),
    input_lin_dgrad_(),
    input_lin_wgrad_(),
    output_lin_fwd_(),
    output_lin_dgrad_(),
    output_lin_wgrad_()
  { }

  void FwdInit(int heads, int embed_dim)
  {
    ++instances_fwd_;
    if (instances_fwd_ > 1) return;
 
    const int input_lin_output_size  = 3 * embed_dim;
    const cublasOperation_t no_trans = CUBLAS_OP_N;
    const cublasOperation_t trans    = CUBLAS_OP_T;
    input_lin_fwd_.CreateStructs(
                                trans,  				// Transpose A?
                                no_trans,  				// Transpose B?
                                input_lin_output_size,  // M
                                1,           			// N
                                embed_dim,   			// K
                                embed_dim,   			// Leading Dim A
                                embed_dim,   			// Leading Dim B
                                input_lin_output_size); // Leading Dim C
    output_lin_fwd_.CreateStructs(
                                trans,                  // Transpose A?
                                no_trans,               // Transpose B?
                                embed_dim,              // M
                                1,  	                // N
                                embed_dim,              // K
                                embed_dim,              // Leading Dim A
                                embed_dim,              // Leading Dim B
                                embed_dim);             // Leading Dim C
  }

  void BwdInit(int heads, int embed_dim)
  {
    ++instances_bwd_;
    if (instances_bwd_ > 1) return;
 
    const int input_lin_output_size  = 3 * embed_dim;
    const cublasOperation_t no_trans = CUBLAS_OP_N;
    const cublasOperation_t trans    = CUBLAS_OP_T;
    input_lin_dgrad_.CreateStructs(
                                no_trans,               // Transpose A?
                                no_trans,               // Transpose B?
                                embed_dim,              // M
                                1,                      // N
                                input_lin_output_size,  // K
                                embed_dim,              // Leading Dim A
                                input_lin_output_size,  // Leading Dim B
                                embed_dim);             // Leading Dim C
    input_lin_wgrad_.CreateStructs(
                                no_trans,               // Transpose A?
                                trans,                  // Transpose B?
                                embed_dim,              // M
                                input_lin_output_size,  // N
                                1,                      // K
                                embed_dim,              // Leading Dim A
                                input_lin_output_size,  // Leading Dim B
                                embed_dim);             // Leading Dim C
    output_lin_dgrad_.CreateStructs(
                                no_trans,               // Transpose A?
                                no_trans,               // Transpose B?
                                embed_dim,              // M
                                1,                      // N
                                embed_dim,              // K
                                embed_dim,              // Leading Dim A
                                embed_dim,              // Leading Dim B
                                embed_dim);             // Leading Dim C
    output_lin_wgrad_.CreateStructs(
                                no_trans,               // Transpose A?
                                trans,                  // Transpose B?
                                embed_dim,              // M
                                embed_dim,              // N
                                1,                      // K
                                embed_dim,              // Leading Dim A
                                embed_dim,              // Leading Dim B
                                embed_dim);             // Leading Dim C
  }

  void FwdDel()
  {
    --instances_fwd_;
    if (instances_fwd_ == 0) {
      input_lin_fwd_.DestroyStructs();
      output_lin_fwd_.DestroyStructs();
    }
  }

  void BwdDel()
  {
    --instances_bwd_;
    if (instances_bwd_ == 0) {
      input_lin_dgrad_.DestroyStructs();
      input_lin_wgrad_.DestroyStructs();
      output_lin_dgrad_.DestroyStructs();
      output_lin_wgrad_.DestroyStructs();
    }
  }

  void FwdChgBatchSize(cublasLtHandle_t lt_handle, uint64_t batch_size)
  {
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(input_lin_fwd_.mat_b_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(input_lin_fwd_.mat_c_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(output_lin_fwd_.mat_b_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(output_lin_fwd_.mat_c_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
 
    int num_results = 0;
    THCublasCheck(cublasLtMatmulAlgoGetHeuristic(lt_handle, 
        		                input_lin_fwd_.op_desc_, 
        						input_lin_fwd_.mat_a_,
        						input_lin_fwd_.mat_b_,
        						input_lin_fwd_.mat_c_,
        						input_lin_fwd_.mat_c_,
        						input_lin_fwd_.preference_,
        						1,
        						&input_lin_fwd_.heuristic_, 
        						&num_results));
    if (num_results == 0) { THCublasCheck(CUBLAS_STATUS_NOT_SUPPORTED); }
    THCublasCheck(input_lin_fwd_.heuristic_.state);
    THCublasCheck(cublasLtMatmulAlgoGetHeuristic(lt_handle, 
        		                output_lin_fwd_.op_desc_, 
        					    output_lin_fwd_.mat_a_,
        					    output_lin_fwd_.mat_b_,
        					    output_lin_fwd_.mat_c_,
        					    output_lin_fwd_.mat_c_,
        					    output_lin_fwd_.preference_,
        					    1,
        					    &output_lin_fwd_.heuristic_, 
        					    &num_results));
    if (num_results == 0) { THCublasCheck(CUBLAS_STATUS_NOT_SUPPORTED); }
    THCublasCheck(output_lin_fwd_.heuristic_.state);
  }

  void BwdChgBatchSize(cublasLtHandle_t lt_handle, uint64_t batch_size)
  {
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(input_lin_dgrad_.mat_b_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(input_lin_dgrad_.mat_c_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(input_lin_wgrad_.mat_a_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(input_lin_wgrad_.mat_b_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(output_lin_dgrad_.mat_b_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(output_lin_dgrad_.mat_c_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(output_lin_wgrad_.mat_a_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(output_lin_wgrad_.mat_b_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
 
    int num_results = 0;
    THCublasCheck(cublasLtMatmulAlgoGetHeuristic(
                                lt_handle, 
        		                input_lin_dgrad_.op_desc_, 
                                input_lin_dgrad_.mat_a_,
                                input_lin_dgrad_.mat_b_,
                                input_lin_dgrad_.mat_c_,
                                input_lin_dgrad_.mat_c_,
                                input_lin_dgrad_.preference_,
                                1,
                                &input_lin_dgrad_.heuristic_, 
                                &num_results));
    if (num_results == 0) { THCublasCheck(CUBLAS_STATUS_NOT_SUPPORTED); }
    THCublasCheck(input_lin_dgrad_.heuristic_.state);
 
    THCublasCheck(cublasLtMatmulAlgoGetHeuristic(
                                lt_handle, 
                                input_lin_wgrad_.op_desc_, 
                                input_lin_wgrad_.mat_a_,
                                input_lin_wgrad_.mat_b_,
                                input_lin_wgrad_.mat_c_,
                                input_lin_wgrad_.mat_c_,
                                input_lin_wgrad_.preference_,
                                1,
                                &input_lin_wgrad_.heuristic_, 
                                &num_results));
    if (num_results == 0) { THCublasCheck(CUBLAS_STATUS_NOT_SUPPORTED); }
    THCublasCheck(input_lin_wgrad_.heuristic_.state);
 
    THCublasCheck(cublasLtMatmulAlgoGetHeuristic(
                                lt_handle, 
        		                output_lin_dgrad_.op_desc_, 
        						output_lin_dgrad_.mat_a_,
        						output_lin_dgrad_.mat_b_,
        						output_lin_dgrad_.mat_c_,
        						output_lin_dgrad_.mat_c_,
        						output_lin_dgrad_.preference_,
        						1,
        						&output_lin_dgrad_.heuristic_, 
        						&num_results));
    if (num_results == 0) { THCublasCheck(CUBLAS_STATUS_NOT_SUPPORTED); }
    THCublasCheck(output_lin_dgrad_.heuristic_.state);
 
    THCublasCheck(cublasLtMatmulAlgoGetHeuristic(
                                lt_handle, 
        		                output_lin_wgrad_.op_desc_, 
        						output_lin_wgrad_.mat_a_,
        						output_lin_wgrad_.mat_b_,
        						output_lin_wgrad_.mat_c_,
        						output_lin_wgrad_.mat_c_,
        						output_lin_wgrad_.preference_,
        						1,
        						&output_lin_wgrad_.heuristic_, 
        						&num_results));
    if (num_results == 0) { THCublasCheck(CUBLAS_STATUS_NOT_SUPPORTED); }
    THCublasCheck(output_lin_wgrad_.heuristic_.state);
  }
  
  // Instance count of the same attention struct so GEMM descriptors can be reused.
  int instances_fwd_;
  int instances_bwd_;

  // GEMM descriptors are only changed when the batch size changes instead of on each
  // instances to save CPU overhead.
  int batches_fwd_;
  int batches_bwd_;

  CublasLtStructs input_lin_fwd_;
  CublasLtStructs input_lin_dgrad_;
  CublasLtStructs input_lin_wgrad_;

  CublasLtStructs output_lin_fwd_;
  CublasLtStructs output_lin_dgrad_;
  CublasLtStructs output_lin_wgrad_;
};

struct EncdecAttnGemmStructs {
  EncdecAttnGemmStructs() :
    instances_fwd_(0),
    instances_bwd_(0),
    batches_q_fwd_(0),
    batches_kv_fwd_(0),
    batches_q_bwd_(0),
    batches_kv_bwd_(0),
    input_lin_q_fwd_(),
    input_lin_q_dgrad_(),
    input_lin_q_wgrad_(),
    input_lin_kv_fwd_(),
    input_lin_kv_dgrad_(),
    input_lin_kv_wgrad_(),
    output_lin_fwd_(),
    output_lin_dgrad_(),
    output_lin_wgrad_()
  { }

  void FwdInit(int heads, int embed_dim)
  {
    ++instances_fwd_;
    if (instances_fwd_ > 1) return;
 
    const int input_lin_kv_output_size = 2 * embed_dim;
    const cublasOperation_t no_trans   = CUBLAS_OP_N;
    const cublasOperation_t trans      = CUBLAS_OP_T;
    input_lin_q_fwd_.CreateStructs(
							    trans,  					// Transpose A?
        						no_trans,  					// Transpose B?
                 				embed_dim,  				// M
                         		1,  						// N
     							embed_dim,  				// K
 								embed_dim,  				// Leading Dim A
 								embed_dim,  				// Leading Dim B
         						embed_dim); 				// Leading Dim C
    input_lin_kv_fwd_.CreateStructs(           
								trans,  					// Transpose A?
                        		no_trans, 					// Transpose B?
        						input_lin_kv_output_size,	// M
                                1,  						// N
           						embed_dim,  				// K
       							embed_dim,  				// Leading Dim A
       							embed_dim,  			   	// Leading Dim B
								input_lin_kv_output_size); 	// Leading Dim C
    output_lin_fwd_.CreateStructs(             
								trans,  					// Transpose A?
			                    no_trans,  					// Transpose B?
                 				embed_dim,  				// M
                         		1,  						// N
     							embed_dim,  				// K
 								embed_dim,                  // Leading Dim A
 								embed_dim,                  // Leading Dim B
         						embed_dim);                 // Leading Dim C
  }

  void BwdInit(int heads, int embed_dim)
  {
    ++instances_bwd_;
    if (instances_bwd_ > 1) return;
 
    const int input_lin_kv_output_size = 2 * embed_dim;
    const cublasOperation_t no_trans   = CUBLAS_OP_N;
    const cublasOperation_t trans      = CUBLAS_OP_T;
    input_lin_q_dgrad_.CreateStructs(
 							    no_trans,                   // Transpose A?
				                no_trans,                   // Transpose B?
                				embed_dim,  				// M
                        		1,  						// N
    							embed_dim,                  // K
								embed_dim,                  // Leading Dim A
        						embed_dim,                  // Leading Dim B
        						embed_dim);                 // Leading Dim C
    input_lin_q_wgrad_.CreateStructs(
								no_trans,  					// Transpose A?
			                    trans,  					// Transpose B?
           				        embed_dim,  				// M
           					    embed_dim,  				// N
            					1,  						// K
								embed_dim,                  // Leading Dim A
        						embed_dim,                  // Leading Dim B
        						embed_dim);                 // Leading Dim C
    input_lin_kv_dgrad_.CreateStructs(
								no_trans,  					// Transpose A?
		                        no_trans,  					// Transpose B?
       			                embed_dim,  				// M
               	                1,  						// N
    							input_lin_kv_output_size,   // K
       							embed_dim,  				// Leading Dim A
								input_lin_kv_output_size,  	// Leading Dim B
               					embed_dim); 				// Leading Dim C
    input_lin_kv_wgrad_.CreateStructs(
                                no_trans,  					// Transpose A?
                                trans,  					// Transpose B?
                       			embed_dim,  				// M
        						input_lin_kv_output_size,  	// N
                   				1,  						// K
       							embed_dim,  				// Leading Dim A
								input_lin_kv_output_size,  	// Leading Dim B
               					embed_dim); 				// Leading Dim C
    output_lin_dgrad_.CreateStructs(
								no_trans,  					// Transpose A?
			                    no_trans,  					// Transpose B?
                				embed_dim,  				// M
                        		1,  						// N
    							embed_dim,  				// K
								embed_dim,  				// Leading Dim A
								embed_dim,  				// Leading Dim B
        						embed_dim); 				// Leading Dim C
    output_lin_wgrad_.CreateStructs(
								no_trans,  					// Transpose A?
  			                    trans,  					// Transpose B?
                				embed_dim,  				// M
                				embed_dim,  				// N
            					1,  						// K
								embed_dim,                  // Leading Dim A
								embed_dim,                  // Leading Dim B
        						embed_dim);                 // Leading Dim C
  }

  void FwdDel()
  {
    --instances_fwd_;
    if (instances_fwd_ == 0) {
      input_lin_q_fwd_.DestroyStructs();
      input_lin_kv_fwd_.DestroyStructs();
      output_lin_fwd_.DestroyStructs();
    }
  }

  void BwdDel()
  {
    --instances_bwd_;
    if (instances_bwd_ == 0) {
      input_lin_q_dgrad_.DestroyStructs();
      input_lin_q_wgrad_.DestroyStructs();
      input_lin_kv_dgrad_.DestroyStructs();
      input_lin_kv_wgrad_.DestroyStructs();
      output_lin_dgrad_.DestroyStructs();
      output_lin_wgrad_.DestroyStructs();
    }
  }

  void FwdChgBatchSizeQ(cublasLtHandle_t lt_handle, uint64_t batch_size)
  {
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(input_lin_q_fwd_.mat_b_,  CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(input_lin_q_fwd_.mat_c_,  CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(output_lin_fwd_.mat_b_,   CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(output_lin_fwd_.mat_c_,   CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
 
    int num_results = 0;
    THCublasCheck(cublasLtMatmulAlgoGetHeuristic(
								lt_handle, 
      		                    input_lin_q_fwd_.op_desc_, 
      							input_lin_q_fwd_.mat_a_,
      							input_lin_q_fwd_.mat_b_,
      							input_lin_q_fwd_.mat_c_,
      							input_lin_q_fwd_.mat_c_,
      							input_lin_q_fwd_.preference_,
      							1,
      							&input_lin_q_fwd_.heuristic_, 
      							&num_results));
    if (num_results == 0) { THCublasCheck(CUBLAS_STATUS_NOT_SUPPORTED); }
    THCublasCheck(input_lin_q_fwd_.heuristic_.state);
    THCublasCheck(cublasLtMatmulAlgoGetHeuristic(
                                lt_handle, 
      		                    output_lin_fwd_.op_desc_, 
      					        output_lin_fwd_.mat_a_,
      					        output_lin_fwd_.mat_b_,
      					        output_lin_fwd_.mat_c_,
      					        output_lin_fwd_.mat_c_,
      					        output_lin_fwd_.preference_,
      					        1,
      					        &output_lin_fwd_.heuristic_, 
      					        &num_results));
    if (num_results == 0) { THCublasCheck(CUBLAS_STATUS_NOT_SUPPORTED); }
    THCublasCheck(output_lin_fwd_.heuristic_.state);
  }

  void FwdChgBatchSizeKv(cublasLtHandle_t lt_handle, uint64_t batch_size)
  {
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(input_lin_kv_fwd_.mat_b_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(input_lin_kv_fwd_.mat_c_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
 
    int num_results = 0;
    THCublasCheck(cublasLtMatmulAlgoGetHeuristic(
                                lt_handle, 
      		                    input_lin_kv_fwd_.op_desc_, 
      					        input_lin_kv_fwd_.mat_a_,
      					        input_lin_kv_fwd_.mat_b_,
      					        input_lin_kv_fwd_.mat_c_,
      					        input_lin_kv_fwd_.mat_c_,
      					        input_lin_kv_fwd_.preference_,
      					        1,
      					        &input_lin_kv_fwd_.heuristic_, 
      					        &num_results));
    if (num_results == 0) { THCublasCheck(CUBLAS_STATUS_NOT_SUPPORTED); }
    THCublasCheck(input_lin_kv_fwd_.heuristic_.state);
  }

  void BwdChgBatchSizeQ(cublasLtHandle_t lt_handle, uint64_t batch_size)
  {
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(input_lin_q_dgrad_.mat_b_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(input_lin_q_dgrad_.mat_c_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(input_lin_q_wgrad_.mat_a_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(input_lin_q_wgrad_.mat_b_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(output_lin_dgrad_.mat_b_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(output_lin_dgrad_.mat_c_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(output_lin_wgrad_.mat_a_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(output_lin_wgrad_.mat_b_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
 
    int num_results = 0;
    THCublasCheck(cublasLtMatmulAlgoGetHeuristic(
                                lt_handle, 
      		                    input_lin_q_dgrad_.op_desc_, 
      					        input_lin_q_dgrad_.mat_a_,
      					        input_lin_q_dgrad_.mat_b_,
      					        input_lin_q_dgrad_.mat_c_,
      					        input_lin_q_dgrad_.mat_c_,
      					        input_lin_q_dgrad_.preference_,
      					        1,
      					        &input_lin_q_dgrad_.heuristic_, 
      					        &num_results));
    if (num_results == 0) { THCublasCheck(CUBLAS_STATUS_NOT_SUPPORTED); }
    THCublasCheck(input_lin_q_dgrad_.heuristic_.state);
 
    THCublasCheck(cublasLtMatmulAlgoGetHeuristic(
                                lt_handle, 
      		                    input_lin_q_wgrad_.op_desc_, 
      					        input_lin_q_wgrad_.mat_a_,
      					        input_lin_q_wgrad_.mat_b_,
      					        input_lin_q_wgrad_.mat_c_,
      					        input_lin_q_wgrad_.mat_c_,
      					        input_lin_q_wgrad_.preference_,
      					        1,
      					        &input_lin_q_wgrad_.heuristic_, 
      					        &num_results));
    if (num_results == 0) { THCublasCheck(CUBLAS_STATUS_NOT_SUPPORTED); }
    THCublasCheck(input_lin_q_wgrad_.heuristic_.state);
 
    THCublasCheck(cublasLtMatmulAlgoGetHeuristic(
                                lt_handle, 
      		                    output_lin_dgrad_.op_desc_, 
      					        output_lin_dgrad_.mat_a_,
      					        output_lin_dgrad_.mat_b_,
      					        output_lin_dgrad_.mat_c_,
      					        output_lin_dgrad_.mat_c_,
      					        output_lin_dgrad_.preference_,
      					        1,
      					        &output_lin_dgrad_.heuristic_, 
      					        &num_results));
    if (num_results == 0) { THCublasCheck(CUBLAS_STATUS_NOT_SUPPORTED); }
    THCublasCheck(output_lin_dgrad_.heuristic_.state);
 
    THCublasCheck(cublasLtMatmulAlgoGetHeuristic(
                                lt_handle, 
      		                    output_lin_wgrad_.op_desc_, 
      					        output_lin_wgrad_.mat_a_,
      					        output_lin_wgrad_.mat_b_,
      					        output_lin_wgrad_.mat_c_,
      					        output_lin_wgrad_.mat_c_,
      					        output_lin_wgrad_.preference_,
      					        1,
      					        &output_lin_wgrad_.heuristic_, 
      					        &num_results));
  	if (num_results == 0) { THCublasCheck(CUBLAS_STATUS_NOT_SUPPORTED); }
  	THCublasCheck(output_lin_wgrad_.heuristic_.state);
  }

  void BwdChgBatchSizeKv(cublasLtHandle_t lt_handle, uint64_t batch_size)
  {
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(input_lin_kv_dgrad_.mat_b_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(input_lin_kv_dgrad_.mat_c_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(input_lin_kv_wgrad_.mat_a_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    THCublasCheck(cublasLtMatrixLayoutSetAttribute(input_lin_kv_wgrad_.mat_b_, CUBLASLT_MATRIX_LAYOUT_COLS, &batch_size, sizeof(batch_size)));
    
    int num_results = 0;
    THCublasCheck(cublasLtMatmulAlgoGetHeuristic(
                                lt_handle, 
      		                    input_lin_kv_dgrad_.op_desc_, 
      					        input_lin_kv_dgrad_.mat_a_,
      					        input_lin_kv_dgrad_.mat_b_,
      					        input_lin_kv_dgrad_.mat_c_,
      					        input_lin_kv_dgrad_.mat_c_,
      					        input_lin_kv_dgrad_.preference_,
      					        1,
      					        &input_lin_kv_dgrad_.heuristic_, 
      					        &num_results));
    if (num_results == 0) { THCublasCheck(CUBLAS_STATUS_NOT_SUPPORTED); }
    THCublasCheck(input_lin_kv_dgrad_.heuristic_.state);
 
    THCublasCheck(cublasLtMatmulAlgoGetHeuristic(
                                lt_handle, 
      		                    input_lin_kv_wgrad_.op_desc_, 
      					        input_lin_kv_wgrad_.mat_a_,
      					        input_lin_kv_wgrad_.mat_b_,
      					        input_lin_kv_wgrad_.mat_c_,
      					        input_lin_kv_wgrad_.mat_c_,
      					        input_lin_kv_wgrad_.preference_,
      					        1,
      					        &input_lin_kv_wgrad_.heuristic_, 
      					        &num_results));
    if (num_results == 0) { THCublasCheck(CUBLAS_STATUS_NOT_SUPPORTED); }
    THCublasCheck(input_lin_kv_wgrad_.heuristic_.state);
  }

  // Instance count of the same attention struct so GEMM descriptors can be reused.
  int instances_fwd_;
  int instances_bwd_;

  // GEMM descriptors are only changed when the batch size changes instead of on each
  // instances to save CPU overhead.
  int batches_q_fwd_;
  int batches_kv_fwd_;
  int batches_q_bwd_;
  int batches_kv_bwd_;

  CublasLtStructs input_lin_q_fwd_;
  CublasLtStructs input_lin_q_dgrad_;
  CublasLtStructs input_lin_q_wgrad_;
  
  CublasLtStructs input_lin_kv_fwd_;
  CublasLtStructs input_lin_kv_dgrad_;
  CublasLtStructs input_lin_kv_wgrad_;

  CublasLtStructs output_lin_fwd_;
  CublasLtStructs output_lin_dgrad_;
  CublasLtStructs output_lin_wgrad_;
};

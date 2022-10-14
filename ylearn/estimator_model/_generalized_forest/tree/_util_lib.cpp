#include <iostream>
#include <cstring>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

typedef Matrix<double,Dynamic,Dynamic,ColMajor > MatrixType;
typedef Matrix<double,Dynamic,Dynamic,RowMajor > MatrixTypeRow;

int eigen_solve(int m,int n,int r, const double *A, const double * B, double *X){
//    cerr <<"m="<<m<<", n="<<n <<", r="<<r <<endl;

    Map<MatrixType> mA=Map<MatrixType>((double*)A,m,n);
    Map<MatrixType> mB=Map<MatrixType>((double*)B,m,r);
    MatrixType mX=mA.bdcSvd(ComputeThinU | ComputeThinV).solve(mB);

    memcpy(X,mX.data(), n * r * sizeof(double));

    return 0;
}

int eigen_solve_r(int m,int n,int r, const double *A, const double * B, double *X){
//    cerr <<"m="<<m<<", n="<<n <<", r="<<r <<endl;

    Map<MatrixTypeRow> mA=Map<MatrixTypeRow>((double*)A,m,n);
    Map<MatrixTypeRow> mB=Map<MatrixTypeRow>((double*)B,m,r);
    MatrixType mX=mA.bdcSvd(ComputeThinU | ComputeThinV).solve(mB);

    memcpy(X,mX.data(), n * r * sizeof(double));

    return 0;
}

int eigen_pinv(int m, int n, const double *A, double*AI){
    Map<MatrixType> mA=Map<MatrixType>((double*)A,m,n);
    MatrixType mInv=mA.completeOrthogonalDecomposition().pseudoInverse();
    memcpy(AI, mInv.data(), m * n * sizeof(double));

    return 0;
}

int inline eigen_pinv_inplace(int m, int n, double *A){
    return eigen_pinv(m, n, A, A);
}

int eigen_pinv_r(int m, int n, const double *A, double*AI){
    Map<MatrixTypeRow> mA=Map<MatrixTypeRow>((double*)A,m,n);
    MatrixTypeRow mInv=mA.completeOrthogonalDecomposition().pseudoInverse();
    memcpy(AI, mInv.data(), m * n * sizeof(double));

    return 0;
}

int inline eigen_pinv_inplace_r(int m, int n, double *A){
    return eigen_pinv_r(m, n, A, A);
}

int init_criterion( /* input */
                    int d_y,int d_tr, int n_samples,
                    const double *y, const double *tr,
                    const double *sample_weight, const long *samples,
                    int start, int end,
                    /* output */
                    double *sum_total,  double *mean_sum,
                    double *sum_tr,double * mean_tr,
                    double *rho, double *grad,
                    double *weighted_n_node_samples,
                    double *sum_rho
){
        // zero outputs
        memset(sum_total, 0, d_y * sizeof(double));
        memset(mean_sum, 0, d_y * sizeof(double));
        memset(sum_tr, 0, d_tr * sizeof(double));
        memset(mean_tr, 0, d_tr * sizeof(double));
        memset(rho, 0, n_samples * sizeof(double));
        memset(grad, 0, d_tr * d_tr * sizeof(double));
        *weighted_n_node_samples = 0.0;
        *sum_rho = 0.0;

        // alloc work memory
        int n_node_samples = end - start;
        int mem_size= n_node_samples * d_y;  // y_node
        mem_size += n_node_samples * d_tr; // tr_node
        mem_size += d_y * d_tr; // ls_coef
        mem_size += d_tr * d_tr; // f_grad
        mem_size += n_node_samples * d_tr; // grad_coef
        mem_size += n_node_samples * d_y; // y_dif
        mem_size += n_node_samples * d_tr; // tr_dif
        mem_size += n_node_samples * d_y; // y_node_cpy

        double *mem_work = new double[mem_size];
        double *y_node = mem_work;
        double *tr_node = y_node + n_node_samples * d_y;
        double *ls_coef = tr_node + n_node_samples * d_tr;
        double *f_grad = ls_coef + d_y * d_tr; // grad in Fortran-contiguous layout
        double *grad_coef = f_grad + d_tr * d_tr;
        double *y_dif = grad_coef + n_node_samples * d_tr;
        double *tr_dif = y_dif + n_node_samples * d_y;
        double *y_node_cpy = tr_dif + n_node_samples * d_tr;


        int p, i, idx, k, k_tr, j_tr;
        double w = 1.0;
        double y_ik, w_y_ik;
        double tr_ik, w_tr_ik;

        // -3 ...
        for(p=start; p<end; p++){
            i = samples[p];
            idx = p - start;

            if(sample_weight != NULL){
                w = sample_weight[i];
            }

            for(k=0; k<d_y; k++){
                y_ik = y[i * d_y + k];
                w_y_ik = w * y_ik;
                y_node[idx + n_node_samples * k] = w_y_ik;
                sum_total[k] += w_y_ik;
                // self.sq_sum_total += w_y_ik * y_ik;
            }

            // compute sum of tr for computing its mean which then gives us tr_dif
            for(k_tr=0; k_tr<d_tr; k_tr++){
                tr_ik = tr[i * d_tr + k_tr];
                w_tr_ik = w * tr_ik;
                tr_node[idx + n_node_samples * k_tr] = w_tr_ik;
                sum_tr[k_tr] += w_tr_ik;
            }
            *weighted_n_node_samples += w;
        }

        // -2: compute the mean of treatment
        for(k_tr=0; k_tr<d_tr; k_tr++){
            mean_tr[k_tr] += sum_tr[k_tr] / *weighted_n_node_samples;
        }

        // -1: compute the mean of outcome
        for(k=0; k<d_y; k++){
            mean_sum[k] += sum_total[k] / *weighted_n_node_samples;
        }

        // 0: compute ls_coef
        eigen_solve(n_node_samples, d_tr, d_y, tr_node, y_node, y_node_cpy);
        for(k_tr=0; k_tr<d_tr; k_tr++){
            for(k=0; k<d_y; k++){
                ls_coef[k_tr + k * d_tr] = y_node_cpy[k_tr + k * n_node_samples];
            }
        }

        // 1: compute tr_dif and y_dif
        double grad_coef_ik=0.0;
        double tr_dif_ik;

        for(p=start; p<end; p++){
            idx = p - start;

            for(k_tr=0; k_tr<d_tr; k_tr++){
                // tr_dif_ik = tr_node[p + n * k_tr] - self.mean_tr[k_tr]
                tr_dif[idx + n_node_samples * k_tr] = (tr_node[idx + n_node_samples * k_tr] - mean_tr[k_tr]);
            }
            for(k=0; k<d_y; k++){
                // TODO: Note that although here n_outputs is not 1, we actually assume it to be 1 to simplify the implementation, see
                // y_dif_ik = y_node[p + n * k] - self.mean_sum[k]
                // y_dif[p + n * k] = y_dif_ik
                // grad_coef_ik += y_dif_ik
                grad_coef_ik = y_node[idx + n_node_samples * k] - mean_sum[k];
                for(k_tr=0; k_tr<d_tr; k_tr++){
                    grad_coef_ik -= tr_dif[idx + n_node_samples * k_tr] * ls_coef[k_tr + k * d_tr];
                }
            }
            // 2: compute grad_coef
            for(k_tr=0; k_tr<d_tr; k_tr++){
                tr_dif_ik = tr_dif[idx + n_node_samples * k_tr];
                grad_coef[idx + n_node_samples * k_tr] = tr_dif_ik * grad_coef_ik;
                for(j_tr=0; j_tr<d_tr; j_tr++){
                    grad[k_tr * d_tr + j_tr] += (tr_dif_ik * tr_dif[idx + n_node_samples * j_tr] / n_node_samples);
                }
            }
        }

        // 3: compute grad
        // TODO: grad does not change after calling this line
        for(k_tr=0; k_tr<d_tr; k_tr++){
            for(j_tr=0; j_tr<d_tr; j_tr++){
                f_grad[k_tr + d_tr * j_tr] = grad[k_tr * d_tr + j_tr];
            }
        }
        for(k_tr=0; k_tr<d_tr; k_tr++){
            f_grad[k_tr + d_tr * k_tr] += 1e-7;
        }

        // 4: compute the inverse of grad
        eigen_pinv_inplace(d_tr, d_tr, f_grad);

        // 5: compute rho
        for(p=start; p<end; p++){
            i = samples[p];
            idx = p - start;

            if(sample_weight != NULL){
                w = sample_weight[i];
            }

            for(k_tr=0; k_tr<d_tr; k_tr++){
                for(j_tr=0; j_tr<d_tr; j_tr++){
                    rho[i] += (f_grad[j_tr + d_tr * k_tr] * grad_coef[idx + n_node_samples * j_tr]);
                }
            }
            *sum_rho += w * rho[i];
        }

    delete[] mem_work;
    return 0;
}

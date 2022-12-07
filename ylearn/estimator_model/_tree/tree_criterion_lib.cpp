
int on_update( /* inputs */
                const double* y, const long* samples, const double* sample_weight,
                int start, int end, int pos, int new_pos,
                /* outputs */
                double *yt_sum_left_, double* y0_sum_left_,
                double *yt_sq_sum_left_, double* y0_sq_sum_left_,
                int *nt_left_, int *n0_left_
        )
{
    double yt_sum_left=0.0, y0_sum_left=0.0;
    double yt_sq_sum_left=0.0, y0_sq_sum_left=0.0;
    int nt_left=0, n0_left=0;
    int p, i;
    double wi, yi;

    if( (new_pos - pos) <= (end - new_pos) ){
        for(p=pos; p<new_pos; p++){
            i = samples[p];
            yi = y[i];
            wi = sample_weight[i];
            if( wi>0.0 ){
                yt_sum_left += yi;
                yt_sq_sum_left += yi*yi;
                nt_left++;
            } else {
                y0_sum_left += yi;
                y0_sq_sum_left += yi*yi;
                n0_left++;
            }
        }
    } else {
        for(p=end - 1; p>new_pos - 1; p--){
            i = samples[p];
            wi = sample_weight[i];
            yi = y[i];
            if( wi>0.0 ){
                yt_sum_left -= yi;
                yt_sq_sum_left -= yi*yi;
                nt_left--;
            } else {
                y0_sum_left -= yi;
                y0_sq_sum_left -= yi*yi;
                n0_left--;
            }
        }
    }

    *yt_sum_left_ = yt_sum_left;
    *y0_sum_left_ = y0_sum_left;
    *yt_sq_sum_left_ = yt_sq_sum_left;
    *y0_sq_sum_left_ = y0_sq_sum_left;
    *nt_left_ = nt_left;
    *n0_left_ = n0_left;
    return 0;
}
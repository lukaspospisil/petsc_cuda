#ifndef THISISKERNEL_H
#define THISISKERNEL_H

__device__ void device_sort_bubble(double *x, int n){
	int i;
	int m=n;
	int mnew;
	double swap;

	while(m > 0){
		/* Iterate through x */
		mnew = 0;
		for(i=1;i<m;i++){
			/* Swap elements in wrong order */
			if (x[i] < x[i - 1]){
				swap = x[i];
				x[i] = x[i-1];
				x[i-1] = swap;
				mnew = i;
			}
	        }
		m = mnew;
	}
}

/* this is a kernel function which computes a projection of a point onto simplex in nD
 *
 * just for curiosity more details: 
 * take K-dimensional vector x[t,t+T,t+2T,...t+(K-1)T] =: p
 * and compute projection
 * P(p) = arg min || p - y ||_2
 * subject to constraints (which define simplex)
 * y_0 + ... y_{K-1} = 1
 * y_i >= 0 for all i=0,...,K-1
 *
 * in practical applications K is much more lower number than T
 * K - number of clusters (2 - 10^2)
 * T - length of time-series (10^5 - 10^9) 
 */ 
__global__ void this_is_kernel(double *x, int T, int K){
	int t = blockIdx.x*blockDim.x + threadIdx.x; /* compute my id */

	if(t<T){ /* maybe we call more than T kernels */
		int k;

		bool is_inside = true;
		double sum = 0.0;
	
		/* control inequality constraints */
		for(k = 0; k < K; k++){ // TODO: could be performed parallely  
			if(x[k*T+t] < 0.0){
				is_inside = false;
			}
			sum += x[k*T+t];
		}

		/* control equality constraints */
		if(sum != 1){ 
			is_inside = false;
		}

		/* if given point is not inside the feasible domain, then do projection */
		if(!is_inside){
			int j,i;
			/* compute sorted x_sub */
			double *y = new double[K];
			double sum_y;
			for(k=0;k<K;k++){
				y[k] = x[k*T+t]; 
			}
			device_sort_bubble(y,K);

			/* now perform analytical solution of projection problem */	
			double t_hat = 0.0;
			i = K - 1;
			double ti;

			while(i >= 1){
				/* compute sum(y) */
				sum_y = 0.0;
				for(j=i;j<K;j++){ /* sum(y(i,n-1)) */
					sum_y += y[j];
				}
				
				ti = (sum_y - 1.0)/(double)(K-i);
				if(ti >= y[i-1]){
					t_hat = ti;
					i = -1; /* break */
				} else {
					i = i - 1;
				}
			}

			if(i == 0){
				t_hat = (sum-1.0)/(double)K; /* uses sum=sum(x_sub) */
			}
    
			for(k = 0; k < K; k++){ // TODO: could be performed parallely  
				/* (*x_sub)(i) = max(*x_sub-t_hat,0); */
				ti = x[k*T+t] - t_hat;	
				if(ti > 0.0){
					x[k*T+t] = ti;
				} else {
					x[k*T+t] = 0.0;
				}
			}
			
			delete y;
		}
		
	}

	/* if t >= T then relax and do nothing */	
}

#endif

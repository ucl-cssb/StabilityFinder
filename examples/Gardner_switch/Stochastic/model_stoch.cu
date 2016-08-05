#define NSPECIES 2
#define NPARAM 4
#define NREACT 4

#define leq(a,b) a<=b
#define neq(a,b) a!=b
#define geq(a,b) a>=b
#define lt(a,b) a<b
#define gt(a,b) a>b
#define eq(a,b) a==b
#define and_(a,b) a&&b
#define or_(a,b) a||b

__device__ double function_1(double a1,double a2){
    return pow(a1, a2);
}


__constant__ int smatrix[]={
	//  S1   , S2
	   -1.0,   0.0,
	    1.0,   0.0,
	    0.0,  -1.0,
	    0.0,   1.0 };

__device__ void hazards(int *y, float *h, float t, int tid){

    h[0] = y[0];
    h[1] = tex2D(param_tex,0,tid)/(1+function_1( y[1], tex2D(param_tex,1,tid)));
    h[2] = y[1];
    h[3] = tex2D(param_tex,2,tid)/(1+function_1( y[0], tex2D(param_tex,3,tid)));
}

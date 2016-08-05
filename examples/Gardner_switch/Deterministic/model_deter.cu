#define NSPECIES 2
#define NPARAM 5
#define NREACT 2

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


struct myFex{
    __device__ void operator()(int *neq, double *t, double *y, double *ydot/*, void *otherData*/){

        int tid = blockDim.x * blockIdx.x + threadIdx.x;


        ydot[0]=tex2D(param_tex,0,tid)*(((tex2D(param_tex,1,tid))/(1+function_1(y[1],tex2D(param_tex,2,tid))))-y[0]);
        ydot[1]=tex2D(param_tex,0,tid)*(((tex2D(param_tex,3,tid))/(1+function_1(y[0],tex2D(param_tex,4,tid))))-y[1]);

    }
};

 struct myJex{
    __device__ void operator()(int *neq, double *t, double *y, int ml, int mu, double *pd, int nrowpd/*, void *otherData*/){
        return;
    }
};
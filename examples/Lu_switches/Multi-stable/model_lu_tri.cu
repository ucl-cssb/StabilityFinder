#define NSPECIES 2
#define NPARAM 8
#define NREACT 2

#define gx tex2D(param_tex,0,tid)

#define kx tex2D(param_tex,1,tid)

#define nXY tex2D(param_tex,2,tid)

#define xXY tex2D(param_tex,3,tid)

#define lXY tex2D(param_tex,4,tid)

#define nXX tex2D(param_tex,5,tid)

#define xXX tex2D(param_tex,6,tid)

#define lXX tex2D(param_tex,7,tid)

__device__ double HS(double x, double xI, double nI, double lI ){
   return 1/(1 + pow(x/xI,nI)) + lI*(1 - 1/(1 + pow(x/xI,nI)) );
}


struct myFex{
    __device__ void operator()(int *neq, double *t, double *y, double *ydot/*, void *otherData*/){

        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        ydot[0] = gx*HS( y[1], xXY, nXY, lXY )*HS( y[0], xXX, nXX, lXX ) - kx*y[0]; //*HS( y[0], xXX, nXX, lXX );
        ydot[1] = gx*HS( y[0], xXY, nXY, lXY )*HS( y[1], xXX, nXX, lXX ) - kx*y[1]; //*HS( y[1], xXX, nXX, lXX );
    }
};

 struct myJex{
    __device__ void operator()(int *neq, double *t, double *y, int ml, int mu, double *pd, int nrowpd/*, void *otherData*/){
        return;
    }
};
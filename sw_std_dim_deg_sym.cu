#define NSPECIES 12
#define NPARAM 11
#define NREACT 20

#define leq(a,b) a<=b
#define neq(a,b) a!=b
#define geq(a,b) a>=b
#define lt(a,b) a<b
#define gt(a,b) a>b
#define eq(a,b) a==b
#define and_(a,b) a&&b
#define or_(a,b) a||b
struct myFex{
    __device__ void operator()(int *neq, double *t, double *y, double *ydot/*, void *otherData*/)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;

        ydot[0]=(1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,1,tid)*y[1])-2.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,4,tid)*__powf(y[0],2))+2.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,5,tid)*y[4])-1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,6,tid)*y[0]))/tex2D(param_tex,0,tid);
        ydot[1]=(-1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,2,tid)*y[1]*y[5])+1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,3,tid)*y[7]))/tex2D(param_tex,0,tid);
        ydot[2]=(1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,1,tid)*y[3])-2.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,4,tid)*__powf(y[2],2))+2.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,5,tid)*y[5])-1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,6,tid)*y[2]))/tex2D(param_tex,0,tid);
        ydot[3]=(-1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,2,tid)*y[3]*y[4])+1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,3,tid)*y[6]))/tex2D(param_tex,0,tid);
        ydot[4]=(1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,4,tid)*__powf(y[0],2))-1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,5,tid)*y[4])-1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,2,tid)*y[3]*y[4])+1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,3,tid)*y[6])-1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,7,tid)*y[8]*y[4])+1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,8,tid)*y[9])-1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,10,tid)*y[4]))/tex2D(param_tex,0,tid);
        ydot[5]=(1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,4,tid)*__powf(y[2],2))-1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,5,tid)*y[5])-1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,2,tid)*y[1]*y[5])+1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,3,tid)*y[7])-1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,7,tid)*y[10]*y[5])+1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,8,tid)*y[11])-1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,10,tid)*y[5]))/tex2D(param_tex,0,tid);
        ydot[6]=(1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,2,tid)*y[3]*y[4])-1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,3,tid)*y[6]))/tex2D(param_tex,0,tid);
        ydot[7]=(1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,2,tid)*y[1]*y[5])-1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,3,tid)*y[7]))/tex2D(param_tex,0,tid);
        ydot[8]=(-1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,7,tid)*y[8]*y[4])+1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,8,tid)*y[9])-1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,9,tid)*y[8]))/tex2D(param_tex,0,tid);
        ydot[9]=(1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,7,tid)*y[8]*y[4])-1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,8,tid)*y[9]))/tex2D(param_tex,0,tid);
        ydot[10]=(-1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,7,tid)*y[10]*y[5])+1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,8,tid)*y[11])-1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,9,tid)*y[10]))/tex2D(param_tex,0,tid);
        ydot[11]=(1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,7,tid)*y[10]*y[5])-1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,8,tid)*y[11]))/tex2D(param_tex,0,tid);

    }
};


 struct myJex{
    __device__ void operator()(int *neq, double *t, double *y, int ml, int mu, double *pd, int nrowpd/*, void *otherData*/){
        return; 
    }
};
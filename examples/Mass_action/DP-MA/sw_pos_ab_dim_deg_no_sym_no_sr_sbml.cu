#define NSPECIES 10
#define NPARAM 13
#define NREACT 20



__constant__ int smatrix[]={
    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    -2.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    -2.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    2.0,    0.0,    0.0,    -1.0,    0.0,    0.0,    0.0,    0.0,
    2.0,    0.0,    0.0,    0.0,    -1.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    -1.0,    0.0,    0.0,    0.0,    -1.0,    1.0,    0.0,    0.0,    0.0,
    0.0,    1.0,    0.0,    0.0,    0.0,    1.0,    -1.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    -1.0,    -1.0,    0.0,    0.0,    1.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    1.0,    1.0,    0.0,    0.0,    -1.0,    0.0,    0.0,
    -1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    -1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    -1.0,    0.0,    -1.0,    0.0,    0.0,    1.0,    0.0,
    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    1.0,    0.0,    1.0,    0.0,    0.0,    -1.0,    0.0,
    0.0,    -1.0,    0.0,    0.0,    -1.0,    0.0,    0.0,    0.0,    0.0,    1.0,
    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    1.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    -1.0,
    0.0,    0.0,    0.0,    0.0,    -1.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    -1.0,    0.0,    0.0,    0.0,    0.0,
};


__device__ void hazards(int *y, float *h, float t, int tid){

    h[0] = tex2D(param_tex,0,tid)*tex2D(param_tex,1,tid)*y[1];
    h[1] = tex2D(param_tex,0,tid)*tex2D(param_tex,11,tid)*y[3];
    h[2] = tex2D(param_tex,0,tid)*tex2D(param_tex,4,tid)*(y[0]*y[0]);
    h[3] = tex2D(param_tex,0,tid)*tex2D(param_tex,4,tid)*(y[2]*y[2]);
    h[4] = tex2D(param_tex,0,tid)*tex2D(param_tex,5,tid)*y[5];
    h[5] = tex2D(param_tex,0,tid)*tex2D(param_tex,5,tid)*y[4];
    h[6] = tex2D(param_tex,0,tid)*tex2D(param_tex,2,tid)*y[1]*y[5];
    h[7] = tex2D(param_tex,0,tid)*tex2D(param_tex,3,tid)*y[6];
    h[8] = tex2D(param_tex,0,tid)*tex2D(param_tex,12,tid)*y[3]*y[4];
    h[9] = tex2D(param_tex,0,tid)*tex2D(param_tex,3,tid)*y[7];
    h[10] = tex2D(param_tex,0,tid)*tex2D(param_tex,6,tid)*y[0];
    h[11] = tex2D(param_tex,0,tid)*tex2D(param_tex,6,tid)*y[2];
    h[12] = tex2D(param_tex,0,tid)*tex2D(param_tex,7,tid)*y[5]*y[3];
    h[13] = tex2D(param_tex,0,tid)*tex2D(param_tex,8,tid)*y[8];
    h[14] = tex2D(param_tex,0,tid)*tex2D(param_tex,9,tid)*y[8];
    h[15] = tex2D(param_tex,0,tid)*tex2D(param_tex,7,tid)*y[4]*y[1];
    h[16] = tex2D(param_tex,0,tid)*tex2D(param_tex,8,tid)*y[9];
    h[17] = tex2D(param_tex,0,tid)*tex2D(param_tex,9,tid)*y[9];
    h[18] = tex2D(param_tex,0,tid)*tex2D(param_tex,10,tid)*y[4];
    h[19] = tex2D(param_tex,0,tid)*tex2D(param_tex,10,tid)*y[5];

}


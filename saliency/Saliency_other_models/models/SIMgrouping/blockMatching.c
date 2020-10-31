#include <stdio.h>
#include "mex.h"
#include "matrix.h"
#include "math.h"

double amax(double a[], int size);
double amin(double a[], int size);

void mexFunction( int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {

/* Declarations */
mxArray *mirroredArray[1], *mirroringrhs[2];
double *plane_values, *mirrored_plane, *mirroringrhsPtr1, *mirroringrhsPtr2;
double *minAD, *ave, *diff;
int i,j,iPad_sz,comp_ind,comp_val,bSide_i,bSide_j,iShiftSz;
int rowLen, colLen, pRowLen, pColLen,sBlock_ind,block_ind, count_i,count_j;
double bSide, shiftSz, pad_sz, AD, blockShft;
double *s;
double *a, *d;
double dims[2];

/* Get matrix x */
plane_values = mxGetPr(prhs[0]);
rowLen = mxGetM(prhs[0]);
colLen = mxGetN(prhs[0]);

/* blocking matching side */
bSide = 3;

/* pad borders symmetrically: */
s          = (double *) mxGetPr(prhs[1]);
shiftSz    = 1;
iShiftSz   = (int) shiftSz;
blockShft  = floor(bSide/2);
pad_sz     = shiftSz+blockShft;

dims[0] = (double) rowLen;
dims[1] = (double) colLen;

if (pad_sz < amin(dims,2)) {
    
    /* pad plane */
    mirroringrhs[0]  = mxCreateDoubleMatrix(rowLen,colLen, mxREAL);
    mirroringrhs[1]  = mxCreateDoubleMatrix(1,1, mxREAL);
    mirroringrhsPtr1 = mxGetPr(mirroringrhs[0]);
    mirroringrhsPtr2 = mxGetPr(mirroringrhs[1]);
    
    for (i=0;i<rowLen;i++) {
        for (j=0;j<colLen;j++) {
            mirroringrhsPtr1[(j*rowLen)+i] = plane_values[(j*rowLen)+i];
        }
    }    
    *mirroringrhsPtr2 = pad_sz;
    
    mexCallMATLAB(1, mirroredArray, 2, mirroringrhs, "mirroring");
    
    mirrored_plane = mxGetPr(mirroredArray[0]);
    pRowLen        = mxGetM(mirroredArray[0]);
    pColLen        = mxGetN(mirroredArray[0]);
    
    iPad_sz = (int) pad_sz;
    minAD   = (double*) malloc(sizeof(double)*pRowLen*pColLen); 
    ave     = (double*) malloc(sizeof(double)*pRowLen*pColLen); 
    diff    = (double*) malloc(sizeof(double)*pRowLen*pColLen); 
    for(i=iPad_sz;i<(pRowLen-iPad_sz);i++) {
        for(j=iPad_sz;j<(pColLen-iPad_sz);j++) {
            minAD[(j*pRowLen)+i] = 255*bSide*bSide;     /* initialize min abs diff to max possible value */
            for (comp_ind = -iShiftSz;comp_ind <= iShiftSz;comp_ind++) {
                comp_val = (j-1)*pRowLen+i+comp_ind;    /* initialize location for matching */
                AD = 0;
                for (bSide_i=-blockShft;bSide_i <= blockShft;bSide_i++) {
                    for (bSide_j=-blockShft;bSide_j <= blockShft;bSide_j++) {
                        sBlock_ind = comp_val+(bSide_j)*pRowLen+bSide_i;
                        block_ind  = comp_val+(bSide_j+1)*pRowLen+bSide_i-comp_ind;
                        AD += fabs(mirrored_plane[sBlock_ind] - mirrored_plane[block_ind]);
                    }                    
                }
                if ((AD < minAD[(j*pRowLen)+i]) || ((comp_ind == 0) && (AD == minAD[(j*pRowLen)+i]))) {
                    minAD[(j*pRowLen)+i] = AD;
                    ave[(j*pRowLen)+i]   = (mirrored_plane[comp_val]+mirrored_plane[(j*pRowLen)+i])/2;
                    diff[(j*pRowLen)+i]  = (mirrored_plane[comp_val]-mirrored_plane[(j*pRowLen)+i])/(pow(2,*s));
                }
            }
        }
    }    
    
    plhs[0] = mxCreateDoubleMatrix(rowLen,floor(colLen/2), mxREAL);
    plhs[1] = mxCreateDoubleMatrix(rowLen,floor(colLen/2), mxREAL);

    /* Get a pointer to the data space in our newly allocated memory */
    a = mxGetPr(plhs[0]);
    d = mxGetPr(plhs[1]);    
    
    /* define association fields: */
    count_i = 0;
    for(i=iPad_sz;i<(pRowLen-iPad_sz);i++) {
        count_j = 0;
        for(j=iPad_sz+1;j<(pColLen-iPad_sz);j+=2) {
            a[count_i+rowLen*count_j] = ave[(j*pRowLen)+i];
            d[count_i+rowLen*count_j] = diff[(j*pRowLen)+i];
            count_j++;
        }
        count_i++;
    }    
    mxDestroyArray(mirroringrhs[0]); mxDestroyArray(mirroringrhs[1]);
    mxDestroyArray(mirroredArray[0]);
    free(minAD); free(ave); free(diff);
}
else {

    plhs[0] = mxCreateDoubleMatrix(1,1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1,1, mxREAL);

    /* Get a pointer to the data space in our newly allocated memory */
    a = mxGetPr(plhs[0]);
    d = mxGetPr(plhs[1]);
    
    *a = 0;
    *d = 0;
}
 
}

double amax(double a[], int size) {
    int i;
    double maxVal = a[0];
    for (i=1; i<size; i++) {
        if (a[i] > maxVal) {
            maxVal = a[i];
        }
    }
    return maxVal;
}

double amin(double a[], int size) {
    int i;
    double minVal = a[0];
    for (i=1; i<size; i++) {
        if (a[i] < minVal) {
            minVal = a[i];
        }
    }
    return minVal;
}

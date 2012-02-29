// Kernel

__global__ void changeHidBias(double *hidBiasD, int *hADData, int *hAD)
{
	hidBiasD[threadIdx.x] += EPSILONHB * (hADData[threadIdx.x] - hAD[threadIdx.x]) / NUMUSERS;
}

__global__ void changeVisBias(double *visBiasD, int *vADData, int *vAD, int *movieCountD)
{
	int offset = blockIdx.y*NUMMOVIES + blockIdx.x*256 + threadIdx.x;
	visBiasD[offset] += EPSILONVB * (vADData[offset] - vAD[offset]) / movieCountD[blockIdx.x*256 + threadIdx.x];
}

__global__void changeWeight(double *wD, int r, int *vADData, int *vAD, int *hADData, int *hAD, int *movieCountD)
{
	int offsetA = r * NUMMOVIES * NUMHIDDEN;
	int offsetB = blockIdx.x * 256 + threadIdx.x;
	int offsetC = NUMHIDDEN * offsetB;
	wD[offsetA + offsetC + blockIdx.y] += EPSILONW * (vADData[r * NUMMOVIES + offsetB] * hADData[blockIdx.y] - vAD[r * NUMMOVIES + offsetB] * hAD[blockIdx.y]) / movieCountD[offsetB];
}

__global__ void consolidateForH(double *vAD, double *wD, double *probH)
{
	__shared__ double product[128];
	int offsetA = blockIdx.x * 256 + threadIdx.x;
	int offsetB = NUMHIDDEN * offsetA;
	int offsetC = blockIdx.y + offsetB;
	int offsetD = NUMMOVIES * NUMHIDDEN;
	double temp;
	temp = vAD[offsetA] * wD[offsetC];
	temp += vAD[NUMMOVIES + offsetA] * wD[offsetD + offsetC];
	temp += vAD[2*NUMMOVIES + offsetA] * wD[2 * offsetD + offsetC];
	temp += vAD[3*NUMMOVIES + offsetA] * wD[3 * offsetD + offsetC];
	temp += vAD[4*NUMMOVIES + offsetA] * wD[4 * offsetD + offsetC];

	__syncthreads();
	if(threadIdx.x > 127)
		product[threadIdx.x-128] = temp;
	__syncthreads();
	if(threadIdx.x < 128)
		product[threadIdx.x] += temp;
	__syncthreads();
	if(threadIdx.x < 32)
		product[threadIdx.x] += product[threadIdx.x + 32] + product[threadIdx.x + 64] + product[threadIdx.x + 96];
	__syncthreads();
	if(threadIdx.x < 8)
		product[threadIdx.x] += product[threadIdx.x + 8] + product[threadIdx.x + 16] + product[threadIdx.x + 24];
	__syncthreads();
	if(threadIdx.x < 2)
		product[threadIdx.x] += product[threadIdx.x + 2] + product[threadIdx.x + 4] + product[threadIdx.x + 6];
	__syncthreads();
	if(threadIdx.x < 1){
		product[0] += product[1];
		probH[70 * blockIdx.y + blockIdx.x] = product[0]
	}
}

__global__ void consolidateForV(double *hAD, double *wD, double *probV, int r){
	probV[r * NUMMOVIES + blockIdx.x * 256 + threadIdx.x] += hAD[blockIdx.y] * wD[NUMHIDDEN * NUMMOVIES * r + NUMHIDDEN * (blockIdx.x * 256 + threadIdx.x) + blockIdx.y];
}

__global__ void perfExp(double *probV){
	int offset = blockIdx.x * NUMMOVIES + blockIdx.y * 256 + threadIdx.y;
	probV[offset] = exp(probV[offset]);
}

__global__ void combineH(double *probH){
	int offset = blockIdx.y * 70 + threadIdx.x;
	if(threadIdx.x < 35)
		probH[offset] += probH[offset + 35]
	__syncthreads();
	if(threadIdx.x < 3)
		probH[offset] += probH[offset + 32]
	__syncthreads();
	if(threadIdx.x < 8)
		probH[offset] += probH[offset + 8] + probH[offset + 16] + probH[offset + 24];
	__syncthreads();
	if(threadIdx.x < 2)
		probH[offset] += probH[offset + 2] + probH[offset + 4] + probH[offset + 6];
	__syncthreads();
	if(threadIdx.x < 1)
		probH[offset] += probH[offset + 1];
}

__global__ void hidBias(double *probH, double *hidBiasD){
	probH[70*threadIdx.x] += hidBiasD[threadIdx.x];
}

__global__ void visBias(double *probV, double *visBiasD){
	int offset = blockIdx.x * NUMMOVIES + blockIdx.y * 256 + threadIdx.y;
	probV[offset] += visBiasD[offset];
}

__global__ void perfSigH(double *probH)
{
	int offset = 70 * threadIdx.x;
	probH[offset] = 1/(1 + exp(-probH[offset]));
}

__global__ void activateH(double *probH, double randomVal, int *hAD){
	if(probH[70 * threadIdx.x] >= randomVal)
		hAD[threadIdx.x] = 1;
}

__global__ void activateV(double *probV, double randomVal, int *vAD, int *vADData){
	int offset = blockIdx.x * 256 + threadIdx.x;
	if(vADData[offset] + vADData[offset + NUMMOVIES + vADData[offset + 2 * NUMMOVIES + vADData[offset + 3 * NUMMOVIES] + vADData[offset + 4*NUMMOVIES])
	{
		double temp = probV[offset];
		temp += probV[NUMMOVIES + offset];
		temp += probV[2 * NUMMOVIES + offset];
		temp += probV[3 * NUMMOVIES + offset];
		temp += probV[4 * NUMMOVIES + offset];
		probV[offset] /= temp;
		probV[NUMMOVIES + offset] /= temp;
		probV[2 * NUMMOVIES + offset] /= temp;
		probV[3 * NUMMOVIES + offset] /= temp;
		probV[4 * NUMMOVIES + offset] /= temp;

		temp = 0;
		if(temp += probV[offset] > randomVal)
			vAD[offset] = 1;
		else if(temp += probV[NUMMOVIES + offset] > randomVal)
			vAD[NUMMOVIES + offset] = 1;
		else if(temp += probV[2 * NUMMOVIES + offset] > randomVal)
			vAD[NUMMOVIES + offset] = 1;
		else if(temp += probV[3 * NUMMOVIES + offset] > randomVal)
			vAD[NUMMOVIES + offset] = 1;
		else
			vAD[NUMMOVIES + offset] = 1;
	}
}

__global__ void activateVPrediction(double *probV, int *vAD){
	int offset = blockIdx.x * 256 + threadIdx.x;
	int rating = 1;
	double max = probV[offset];

	if(probV[offset + NUMMOVIES] > max)
	{
		max = probV[offset + NUMMOVIES];
		rating = 2;
	}

	if(probV[offset + 2 * NUMMOVIES] > max)
	{
		max = probV[offset + 2 * NUMMOVIES];
		rating = 3;
	}

	if(probV[offset + 3 * NUMMOVIES] > max)
	{
		max = probV[offset + 3 * NUMMOVIES];
		rating = 4;
	}

	if(probV[offset + 4 * NUMMOVIES] > max)
		rating = 5;
	vAD[offset] = rating;
}
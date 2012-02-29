// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "list.h"

// includes, project
#include <cutil.h>

// includes, kernels
#include <RBM_kernel.cu>

#define NUMRATINGS 100480507
#define WITHHOLD 1408395
#define NUMUSERS 480179
#define NUMMOVIES 17920
// Number of movies is actually 17,770. Caught in activateV, as vADData[i] == 0 for all i > 17770.

// The following constants can be defined per desired fit and annealing schedule.
#define NUMEPOCHS
#define TNUM
#define NUMHIDDEN
#define EPSILONVB
#define EPSILONHB
#define EPSILONW

int main(int argc, char *argv[])
{

// CUDA Execution Configuration Parameters
dim3 bE, tE;
tE.y = 1;
size_t nS;

// Ancillary Variables - Model
double randomVal, tRating;
int tMovie, tUser;
List* users = (List*) malloc(NUMUSERS * sizeof(List));
ListE data;
int *movieCount = calloc(NUMMOVIES, sizeof(int));

// Initialize Lists
for(int a=0; a<NUMUSERS; a++)
	listStart(&users[a]);

// Read Ratings from Binary File
FILE *rF = fopen("ratings.bin", "rb");
for(int a=0; a<NUMRATINGS; a++)
{
	fread(&(tMovie), sizeof(tMovie), 1, rF);
	fread(&(tUser), sizeof(tUser), 1, rF);
	fread(&(tRating), sizeof(tRating), 1, rF);
	listAddNext(&users[tUser], tMovie, (tRating-1));
	movieCount[tMovie]++;
}
fclose(rF);

// Number of Times Each Movie was Rated, Array on Device.
int *movieCountD;
cudaMalloc((void**)&movieCountD, NUMMOVIES * sizeof(int));
cudaMemcpy(movieCount, movieCountD, NUMMOVIES * sizeof(int), cudaMemcpyHostToDevice);

// Activation of Visible Nodes, Array on Host.
int visAct[5 * NUMMOVIES];

double *wD, *probH, *probV, *visBiasD, *hidBiasD;
int *hAD, *hADData, *vAD, *vADData;

// Weights Between Visible and Hidden Nodes.
cudaMalloc((void**)&wD, 5 * NUMMOVIES * NUMHIDDEN * sizeof(double));

// Activation Probabilities.
cudaMalloc((void**)&probH, 70 * NUMHIDDEN * sizeof(double));
cudaMalloc((void**)&probV, 5 * NUMMOVIES * sizeof(double));

// Node Biases.
cudaMalloc((void**)&visBiasD, 5 * NUMMOVIES * sizeof(double));
cudaMalloc((void**)&hidBiasD, NUMHIDDEN * sizeof(double));

// Activation States for Data and Simulation.
cudaMalloc((void**)&hAD, NUMHIDDEN * sizeof(int));
cudaMalloc((void**)&hADData, NUMHIDDEN * sizeof(int));
cudaMalloc((void**)&vAD, 5 * NUMMOVIES * sizeof(int));
cudaMalloc((void**)&vADData, 5 * NUMMOVIES * sizeof(int));

//							//
//							//
//	Initialize Biases and Weights on Device		//
//							//
//							//

for(int nE=0; nE<NUMEPOCHS; nE++)
{
	for(int tCase=0; tCase<NUMUSERS; tCase++)
	{
		// Read User Row and Spread Indices to Full Length
		memset(visAct, 0, 5 * NUMMOVIES * sizeof(int));
		data = *users[tCase].head;
		do{
			visAct[(int)(data.rating * NUMMOVIES + data.secondary)] = 1;
			data = *data.next;
		} while(data.next != NULL);
		visAct[(int)(data.rating * NUMMOVIES + data.secondary)] = 1;

		cudaMemcpy(visAct, vAD, 5 * NUMMOVIES * sizeof(int), cudaMemcpyHostToDevice);

		// Compute Visible Node Activation Product with Weights in Parallel.
		bE.x = 70;
		bE.y = NUMHIDDEN;
		tE.x = 256;
		nS = 128 * sizeof(double);
		consolidateForH<<<bE,tE,nS>>>(vAD, wD, probH);

		// Combine the Results from Each Row of Blocks.
		bE.x = 1;
		tE.x = 70;
		nS = 70 * sizeof(double);
		combineH<<<bE,tE,nS>>>(probH);

		// Add Biases
		bE.x = 1;
		tE.x = NUMHIDDEN;
		bE.y = 1;
		hidBias<<<bE,tE>>>(probH, hidBiasD);

		// Sigmoid
		perfSigH<<<bE,tE>>>(probH);

		// Generate Random Number to Determine Node Activations
		srand((unsigned int) time(NULL));
		randomVal = (double) rand()/((double) RAND_MAX);
		activateH<<<bE,tE>>>(probH, randomVal, hAD);

		// Current Activation States are Directly from Data Interaction with Model
		cudaMemcpy(vAD, vADData, 5 * NUMMOVIES * sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(hAD, hADData, NUMHIDDEN * sizeof(int), cudaMemcpyDeviceToDevice);

		for(int a=0; a<TNUM; a++)
		{
			// Gibbs Step (Visible).
			// Compute Hidden Node Activation Product with Weights in Parallel.
			bE.x = 70;
			tE.x = 256;
			bE.y = NUMHIDDEN;

			// Initialize all probV to 0. This is done automatically for hidden
			// nodes in consolidateForH. It is not possible in this case, because
			// the execution configuration for blocks is limited to two dimensions,
			// so that an additional dimension to represent the five different ratings
			// is not possible.

			cudaMemset(probV,0,5 * NUMMOVIES * sizeof(double));

			for(int r=0; r<5; r++)
				consolidateForV<<<bE,tE>>>(hAD, wD, probV, r);
			// Add Biases
			visBias<<<bE,tE>>>(probV, visBiasD);

			// Perform Exponent. - Same as above?
			perfExp<<<bE,tE>>>(probV);

			// Divide out Fractional Probabilities.
			// Determine Which Node Should be Activated
			// IFF Movie Was Rated by User in Training Set.
			bE.y = 1;
			srand((unsigned int) time(NULL));
			randomVal = (double) rand()/((double) RAND_MAX);
			cudaMemset(vAD,0,5 * NUMMOVIES * sizeof(int));
			activateV<<<bE,tE>>>(probV, randomVal, vAD, vADData);

			// Gibbs Step (Hidden).
			// Compute Visible Node Activation Product with Weights in Parallel.
			bE.x = 70;
			bE.y = NUMHIDDEN;
			tE.x = 256;
			nS = 128 * sizeof(double);
			consolidateForH<<<bE,tE,nS>>>(vAD, wD, probH);

			// Combine the Results from Each Row of Blocks.
			bE.x = 1;
			tE.x = 70;
			nS = 32 * sizeof(double);
			combineH<<<bE,tE,nS>>>(probH);

			// Add Biases, Perform Sigmoid.
			bE.x = 1;
			tE.x = NUMHIDDEN;
			bE.y = 1;
			hidBias<<<bE,tE>>>(probH, hidBiasD);
			perfSigH<<<bE,tE>>>(probH);


			// Generate Random Number to Determine Node Activations
			srand((unsigned int) time(NULL));
			randomVal = (double) rand()/((double) RAND_MAX);
			activateH<<<bE,tE>>>(probH, randomVal, hAD);

		}

		// Update Hidden Bias.
		changeHidBias<<<bE,tE>>>(hidBiasD, hADData, hAD);
		bE.x = 70;
		tE.x = 256;
		bE.y = 5;

		// Update Visible Bias.
		changeVisBias<<<bE,tE>>>(visBiasD, vADData, vAD, movieCountD);

		// Update Weights.
		bE.y = NUMHIDDEN;
		for(int r=0; r<5; r++)
			changeWeight<<<bE,tE>>>(wD, r, vADData, vAD, hADData, hAD, movieCountD);
	}
}

// Model is Ready!

// Read Quiz Set from Binary File.
rF = fopen("quiz.bin", "rb");

// Ancillary Variables - Prediction Generation
List* usersP = (List*) malloc(NUMUSERS * sizeof(List));

cudaMalloc((void**)&predD, sizeof(int));

for(int a=0; a<NUMUSERS; a++)
	listStart(&usersP[a]);

for(int a=0; a<WITHHOLD; a++)
{
	fread(&(tMovie), sizeof(tMovie), 1, rF);
	fread(&(tUser), sizeof(tUser), 1, rF);
	fread(&(tRating), sizeof(tRating), 1, rF);
	listAddNext(&usersP[tUser], tMovie, tRating);
}

fclose(rF);

*rF = fopen("predictions.bin", "wb");


for(int a=0; a<NUMUSERS; a++)
{
	if(usersP[a].size != 0)
	{
		memset(visAct, 0, 5 * NUMMOVIES);
		data = *users[a].head;
		do{
			visAct[(int)(data.rating * NUMMOVIES + data.secondary)] = 1;
			data = *data.next;
		} while(data.next != NULL);
		visAct[(int)(data.rating * NUMMOVIES + data.secondary)] = 1;
		cudaMemcpy(visAct, vAD, NUMMOVIES * 5 * sizeof(int), cudaMemcpyHostToDevice);

		// Compute Visible Node Activation Product with Weights in Parallel.
		bE.x = 70;
		bE.y = NUMHIDDEN;
		tE.x = 256;
		nS = 128 * sizeof(double);
		consolidateForH<<<bE,tE,nS>>>(vAD, wD, probH);

		// Combine the Results from Each Row of Blocks.
		bE.x = 1;
		tE.x = 70;
		nS = 32 * sizeof(double);
		combineH<<<bE,tE,nS>>>(probH);

		// Add Biases, Perform Sigmoid.
		bE.x = 1;
		tE.x = NUMHIDDEN;
		bE.y = 1;
		hidBias<<<bE,tE>>>(probH, hidBiasD);
		perfSigH<<<bE,tE>>>(probH);


		// Generate Random Number to Determine Node Activations
		srand((unsigned int) time(NULL));
		randomVal = (double) rand()/((double) RAND_MAX);
		activateH<<<bE,tE>>>(probH, randomVal, hAD);

		// Compute Hidden Node Activation Product with Weights in Parallel.
		bE.x = 70;
		tE.x = 256;
		bE.y = NUMHIDDEN;

		// Initialize all probV to 0. This is done automatically for hidden
		// nodes in consolidateForH. It is not possible in this case, because
		// the execution configuration for blocks is limited to two dimensions,
		// so that an additional dimension to represent the five different ratings
		// is not possible.

		cudaMemset(probV,0,5 * NUMMOVIES * sizeof(double));

		for(int r=0; r<5; r++)
			consolidateForV<<<bE,tE>>>(hAD, wD, probV, r);
		// Add Biases
		visBias<<<bE,tE>>>(probV, visBiasD);

		// Perform Exponent.
		perfExp<<<bE,tE>>>(probV);

		// Use probV to determine which node should
		// be sctivated for each movie. Selected by
		// highest probability in this version. Also
		// could use expected value, but a different
		// array than vAD would need to be used, as
		// output would be of type  NOTE: Only the
		// first NUMMOVIES elements of vAD are used
		// by this function.
 
		bE.y = 1;
		cudaMemset(vAD,0,5 * NUMMOVIES * sizeof(int));
		activateVPrediction<<<bE,tE>>>(probV, vAD);

		cudaMemcpy(vAD, visAct, 5 * NUMMOVIES * sizeof(int), cudaMemcpyDeviceToHost);

		data = *usersP[a].head;
		do{
			fwrite(&(data.secondary), sizeof(int), 1, rF);
			fwrite(&(a), sizeof(int), 1, rF);
			fwrite(visAct[data.secondary], sizeof(int), 1, rF);
			data = *data.next;
		} while(data.next != NULL);
		data.rating = visAct[data.secondary];

		fwrite(&(data.secondary), sizeof(int), 1, rF);
		fwrite(&(a), sizeof(int), 1, rF);
		fwrite(visAct[data.secondary], sizeof(int), 1, rF);
	}
}

fclose(rF);

return 0;
}
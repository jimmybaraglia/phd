phd
===

My PhD works

#include "StdAfx.h"
#include "LSM.h"


LSM::LSM(int inputs, int output, int neurons)
{
  nbInputs = inputs;
	nbOutputs = output;
	nbNeurons = neurons;
	this->In = new matrix<float>(1,inputs);
	this->IL = new matrix<float>(inputs,neurons);
	connectInputs(40);

	this->Lsm = new matrix<float>(1,neurons);
	this->LsmT = new matrix<float>(1,neurons);
	
	srand(time(NULL));
	for(int cpt = 0; cpt < Lsm->size2(); cpt++)
	{
		int rand1;
		rand1 = rand()%100;
			if(rand1 > 50)
				rand1 = -1;
			else
				rand1 = 1;

		Lsm[0](0,cpt) = 0;
		LsmT[0](0,cpt) = rand1;
	}
	
	this->LL = new matrix<float>(neurons, neurons);
	initReservoir();
	this->LO = new matrix<float>(neurons,output);
	connectOutputs(10);
	this->Out = new matrix<float>(1,output);
}


LSM::~LSM(void)
{

}

void LSM::connectInputs(int nbCon)
{
	int conn, nr;
	srand(time(NULL));

	for(int cpt = 0; cpt < nbNeurons; cpt++)
	{
		for(int cpt2 = 0; cpt2 < nbInputs; cpt2++)
		{
			IL[0](cpt2,cpt) = 0;
		}
	}

	for(int cpt2 = 0; cpt2 < nbInputs; cpt2++)
	{
		conn = 0;
		while(conn < nbCon)
		{
			nr = rand()%nbNeurons;
			if(IL[0](cpt2, nr) == 0)
			{
				IL[0](cpt2, nr) = 1.;
				conn ++;
			}
		}
	}
}

void LSM::connectOutputs(int nbCon)
{
	int conn, nr;
	srand(time(NULL));

	for(int cpt = 0; cpt < nbNeurons; cpt++)
	{
		conn = 0;
		for(int cpt2 = 0; cpt2 < nbOutputs; cpt2++)
		{
			LO[0](cpt,cpt2) = 0;
		}
	}

	for(int cpt2 = 0; cpt2 < nbOutputs; cpt2++)
	{
		conn = 0;
		while(conn < 8)
		{
			nr = rand()%nbNeurons;
			if(LO[0](nr,cpt2) == 0)
			{
				LO[0](nr,cpt2) = rand()%100;
				LO[0](nr,cpt2) /= 100.;
				conn ++;
			}
		}
	}
}

void LSM::initReservoir()
{
	float rand2; 
	float rand3;
	srand(time(NULL));
	for(int cpt = 0; cpt < nbNeurons; cpt++)
	{
		for(int cpt2 = 0; cpt2 < nbNeurons; cpt2++)
		{
			rand2 = rand()%(nbNeurons);
			if(rand2 < ((nbNeurons/10.)-(nbNeurons/11.)))
			{
				rand2 = 1;
			}
			else
			{
				rand2 = 0;
			}

			rand3 = rand()%100;
			rand3 /= 100.;
			rand3 = 1;
			
			LL[0](cpt,cpt2) = rand2*rand3;
		}
		cout<<endl;
	}
}

void LSM::setInput(matrix<float>* in)
{
	if(in->size1() == In->size1())
	{
		for (unsigned i = 0; i < In->size2(); ++ i)
		{
			In[0](0,i) = in[0](0,i);
		}
	}
	else{
		cout<<"Matrix Length error"<<endl;
	}

	calcOutput();
}

void LSM::newInput()
{
	nbInputs++;
	matrix<float>* newM = new matrix<float>(1,nbInputs);
	matrix<float>* newWM = new matrix<float>(nbInputs, nbOutputs);
	for(int i = 0; i < nbInputs; i++)
	{
		for(int j = 0; j < nbOutputs; j++)
		{
			if(i == nbInputs-1)
			{
				newWM[0](i,j) = 0;
			}
			else
			{
				newWM[0](i,j) = IL[0](i,j);
			}
		}
	}
	In = newM;
	IL = newWM;
}

float LSM::dogfunc(float distance, float a){
	return(exp(-a*(2*distance*distance)));
}

double LSM::activation(float input)
{
	float value;
	value = tanh(input);
	if(value < 0.5)
		return 0;
	return value;
}

void LSM::calcOutput()
{
	matrix<float> temp(Out[0].size1(), Out[0].size2());
	matrix<float> temp2(Lsm[0].size1(), Lsm[0].size2());
	matrix<float> tempLsm(Lsm[0].size1(), Lsm[0].size2());
	matrix<float>* temp3 = new matrix<float>(Lsm[0].size1(), Lsm[0].size2());
	matrix<float>* temp4 = new matrix<float>(Out[0].size1(), Out[0].size2());


	for(int cpt = 0; cpt < temp2.size2(); cpt++)
	{
		tempLsm(0,cpt) = LsmT[0](0,cpt) * Lsm[0](0,cpt);
	}
	temp2 = prod(tempLsm, LL[0]);

	temp = prod(In[0], IL[0]);
	for(int cpt = 0; cpt < Lsm->size2(); cpt ++)
	{
		Lsm[0](0,cpt) = activation(temp(0,cpt) + temp2(0,cpt));
//		cout<<Lsm[0](0,cpt)<<endl;
	}
//	cout<<"-------------------------------------------------------"<<endl;
	
	
	temp4[0] = prod(Lsm[0], LO[0]);
	for(int cpt = 0; cpt < Out->size2(); cpt ++)
	{
		Out[0](0,cpt) = activation(temp4[0](0,cpt));
		cout<<Out[0](0,cpt)<<"   /   ";
	}
	cout<<endl;

	displayNeurons();

	/*
	for(int cpt = 0; cpt < Out->size2(); cpt++)
	{
		cout<<Out[0](0,cpt)<<endl;
	}
	cout<<endl;*/
}

void LSM::displayNeurons()
{
	int w, h;

	w = (int)sqrt((float)Lsm->size2());
	h = (int)(((float)Lsm->size2()/(float)w)+1);

	IplImage* nAct = cvCreateImage(cvSize(20*w,20*h), 8, 3);
	cvSet(nAct, cvScalar(0,0,0));
	
	for(int cpt1 = 0; cpt1 < w; cpt1 ++)
	{
		for(int cpt2 = 0; cpt2 < h; cpt2++)
		{
			if(Lsm->size2() > cpt1 + (cpt2*w))
			{
				int x1, x2, y1, y2;
				CvScalar color;
				x1 = cpt1*20-10;
				y1 = cpt2*20-10;

				if(LsmT[0](0,cpt1 + (cpt2*w)) > 0)
					color = cvScalar(Lsm[0](0,cpt1 + (cpt2*w))*255, Lsm[0](0,cpt1 + (cpt2*w))*255,0);
				else
					color = cvScalar(0, Lsm[0](0,cpt1 + (cpt2*w))*255, Lsm[0](0,cpt1 + (cpt2*w))*255);

				cvDrawCircle(nAct, cvPoint(x1,y1), 5, color,10,8,0);
			}
		}
	}
	cvShowImage("neural activity", nAct);
	int c=cvWaitKey(10);
	cvReleaseImage(&nAct);
}

int determinant_sign(const permutation_matrix<std ::size_t>& pm)
{
    int pm_sign=1;
    std::size_t size = pm.size();
    for (std::size_t i = 0; i < size; ++i)
        if (i != pm(i))
            pm_sign *= -1.0; // swap_rows would swap a pair of rows here, so we change sign
    return pm_sign;
}

double puis(int i, int j){
	double out = -1.00;
	for(int k = 0; k < (i+j)+1; k++)
		out *= -1.00;
	return out;
}
 

double determinant(matrix<float> input)
{
  double det;
  if(input.size1() == 2 && input.size2() == 2)
	  det = input(0,0)*input(1,1)-input(0,1)*input(1,0);
  else if(input.size1() == 1 && input.size2() == 1)
	  det = input(0,0);
  else
  {
	det = 0;
	int cpt = 1;
	for (unsigned i = 0; i < input.size1 (); ++ i)
	{
		matrix<double> temp(input.size1()-1, input.size2()-1);
		int x, y;
		x = y = 0;
		for (unsigned k = 0; k < input.size1 (); ++ k)
		{
			for (unsigned l = 0; l < input.size2 (); ++ l)
			{
				if(i != k && 0 != l)
				{
					temp(x,y) = input(k,l);
					y++;
				}
			}
			if(i!=k)
			{
				x++;
				y = 0;
			}
		}
		det += puis(cpt,1)*input(i,0)*determinant(temp);
		cpt++;
	}
  }

  return det;
} 

bool InvertMatrix (matrix<float> input, matrix<float>& inverse) {

 	double det = determinant(input);
	if(det == 0)
		return false;

	if(input.size1() == 1 && input.size2() == 1){
		inverse(0,0) = (1./det);
		return true;
	}

	for (unsigned i = 0; i < input.size1 (); ++ i)
	{
        for (unsigned j = 0; j < input.size2 (); ++ j)
		{
			matrix<double> temp(input.size1()-1, input.size2()-1);
			int x, y;
			x = y = 0;
			for (unsigned k = 0; k < input.size1 (); ++ k)
			{
				for (unsigned l = 0; l < input.size2 (); ++ l)
				{
					if(i != k && j != l)
					{
						temp(x,y) = input(k,l);
						y++;
					}
				}
				if(i!=k)
				{
					x++;
					y = 0;
				}
			}
			inverse(i,j) = determinant(temp)*puis(i,j);
		}
	}
	inverse = trans(inverse);
	inverse *= 1./det;
 	return true;
 }

void LSM::atanh(matrix<float>* in, matrix<float>* out)
{
	for (unsigned i = 0; i < in->size1 (); ++ i)
	{
        for (unsigned j = 0; j < in->size2 (); ++ j)
		{
			double X = in[0](i,j);
			if(X < 1)
				out[0](i,j) = log((1. + X) / (1 - X)) / 2;
			else
				out[0](i,j) = DBL_MAX;
		}
	}
}

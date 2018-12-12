#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

//---------------------------------------------------------------------------
// class to read training data from a file
class TrainingData
{
public:
	TrainingData(const string& filename);
	bool isEof(void) { return trainingDataFile.eof(); }

	void getDiagram(vector<float> & diagram);

	// returns the number of input values read from the file :

	size_t getNextInputs(vector <float> &inputVal);
	size_t getTargetOutputs(vector<float> & middleVal1);

private:
	ifstream trainingDataFile;
};

void TrainingData::getDiagram(vector<float> & diagram)
{
	string line;
	string label;

	getline(trainingDataFile, line);
	stringstream ss(line);

	ss >> label;
	if (this->isEof() || label.compare("diagram: ") != 0)
	{
		abort();
	}

	while (!ss.eof())
	{
		size_t i;

		ss >> i;
		diagram.push_back(i);
	}
	return;
}

TrainingData::TrainingData(const string& filename) :
	trainingDataFile(filename)
{
}

size_t TrainingData::getNextInputs(vector<float> &inputVal)
{
	inputVal.clear();

	string line;
	getline(trainingDataFile, line);
	stringstream ss(line);

	string label;
	ss >> label;

	if (label.compare("in:") == 0)
	{
		float oneValue;
		while (ss >> oneValue)
		{
			inputVal.push_back(oneValue);
		}
	}
	return inputVal.size();

}

size_t TrainingData::getTargetOutputs(vector<float> &targetOutputVal)
{
	targetOutputVal.clear();

	string line;
	getline(trainingDataFile, line);
	stringstream ss(line);

	string label;
	ss >> label;
	if (label.compare("out:") == 0)
	{
		float oneValue;
		while (ss >> oneValue)
		{
			targetOutputVal.push_back(oneValue);
		}
	}
	return targetOutputVal.size();
}

//---------------------------------------------------------------------------
//
struct NeuronDistance
{
	float weight;
	float deltaWeight;
};

class Neuron;
typedef vector<Neuron*> Layer; // layer[number of layer =i][number of neurons=j]

class Neuron
{
public:
	Neuron(size_t numberNeuronnextlayer, size_t myIndex);
	void setOutputVal(float value) { OutputVal = value; }
	float getOutputVal() const { return OutputVal; }
	void input(const Layer &prevLayer);
	void calcOutputGradients(double middle1);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);

private:
	static float eta;   // [0.0...1.0] overall training rate
	static float alpha; // [0.0...n] last weight change multiplier
	static float activationFunction(float Y);
	static float activationFunctionDerivative(float Y);
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	float sumDow(const Layer &nextLayer) const;
	float passAmount;
	vector<NeuronDistance> outputWeight;
	size_t myIndex;
	float gradient;
	float OutputVal;
};

float Neuron::eta = 0.15; // net overall learning rate
float Neuron::alpha = 0.5; // last delta weight multiplier

void Neuron::updateInputWeights(Layer &prevLayer)
{
	// the weights to be updated are in NeuronDistance in the preceding neurons
	for (size_t q = 0; q < prevLayer.size(); ++q)
	{
		Neuron* neuron = prevLayer[q];
		float oldDeltaWeight = neuron->outputWeight[myIndex].deltaWeight;

		// individual input magnified by gradient and training rate
		float newDeltaWeight = eta * neuron->getOutputVal() * gradient + alpha * oldDeltaWeight;

		neuron->outputWeight[myIndex].deltaWeight = newDeltaWeight;
		neuron->outputWeight[myIndex].weight += newDeltaWeight;
	}
}

float Neuron::sumDow(const Layer &nextLayer) const
{
	float sum = 0.0;
	// sum error contribution at the nodes

	for (size_t w = 0; w < nextLayer.size() - 1; ++w)
	{
		sum += outputWeight[w].weight * nextLayer[w]->gradient;
	}
	return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	float dow = sumDow(nextLayer);
	gradient = dow * Neuron::activationFunctionDerivative(passAmount);
}

void Neuron::calcOutputGradients(double middle1)
{
	float diff = middle1 - passAmount;
	gradient = diff * Neuron::activationFunctionDerivative(passAmount);
}

float Neuron::activationFunction(float Y)
{
	return tanh(Y);
}

float Neuron::activationFunctionDerivative(float Y)
{
	return (1 - (tanh(Y)*tanh(Y)));
}

void Neuron::input(const Layer &prevLayer)
{
	float sum = 0.0;

	// sum the previous layer output and include bias neuron
	for (size_t e = 0; e < prevLayer.size(); ++e)
	{
		sum += prevLayer[e]->getOutputVal() * prevLayer[e]->outputWeight[myIndex].weight;
	}

	OutputVal = Neuron::activationFunction(sum);
}

Neuron::Neuron(size_t numberNeuronnextlayer, size_t Index) :
	OutputVal(0.0)
{
	for (int a = 0; a < numberNeuronnextlayer; a++)
	{
		NeuronDistance nuDist
		outputWeight.push_back(nuDist);
		outputWeight.back().weight = randomWeight();
	}
	myIndex = Index;
}

//---------------------------------------------------------------------------
//
class Neural
{
public:
	Neural(const vector <float> &diagram);
	void input(vector<float> &inputVal);
	void middle1(vector<float> &middleVal1);
	//void middle2(vector<double> &middleVal2) {};
	void output(vector<float>& outputVal) const;
	float getrecentAverageError(void) const { return recentAverageError; }

private:
	vector<Layer> numberLayer;
	float error;
	float recentAverageError;
	float recentAverageSmoothingFactor;
};

void Neural::output(vector<float>& outputVal) const
{
	outputVal.clear();
	for (size_t i = 0; i < numberLayer.back().size() - 1; ++i)
	{
		outputVal.push_back(numberLayer.back()[i]->getOutputVal());
	}
}

void Neural::middle1(vector<float>& middleVal1)
{
	// Calculate the RMS error
	Layer &outputLayer = numberLayer.back();
	error = 0.0;
	for (size_t k = 0; k < outputLayer.size() - 1; ++k)
	{
		float diff = middleVal1[k] - outputLayer[k]->getOutputVal();
		error += diff * diff;
	}
	error /= outputLayer.size() - 1;
	error = sqrt(error);

	// implement a recent average measurement
	recentAverageError = (recentAverageError*recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0);

	// Calculate the output layer gradient
	for (size_t u = 0; u < outputLayer.size() - 1; ++u)
	{
		outputLayer[u]->calcOutputGradients(middleVal1[u]);
	}

	// Calculate the hidden layer gradient
	for (size_t layerNum = numberLayer.size() - 2; layerNum > 0; --layerNum)
	{
		Layer &hiddenLayer = numberLayer[layerNum];
		Layer &nextLayer = numberLayer[layerNum + 1];

		for (size_t i = 0; i < hiddenLayer.size(); ++i)
		{
			hiddenLayer[i]->calcHiddenGradients(nextLayer);
		}
	}

	// For all layers from outputs to first hidden layer
	// update weights
	for (size_t layerNum = numberLayer.size() - 1; layerNum > 0; --layerNum)
	{
		Layer &layer = numberLayer[layerNum];
		Layer &prevLayer = numberLayer[layerNum - 1];

		for (size_t r = 0; r < layer.size() - 1; ++r)
		{
			layer[r]->updateInputWeights(prevLayer);
		}
	}
}

void Neural::input(vector<float> &inputVal)
{
	assert(inputVal.size() == numberLayer[0].size() - 1);

	// passing the input values to individual neurons
	for (size_t i = 0; i < inputVal.size(); ++i)
	{
		numberLayer[0][i]->setOutputVal(inputVal[i]);
	}

	// Forward propagation
	for (size_t l = 1; l < numberLayer.size(); ++l)
	{
		Layer &prevLayer = numberLayer[l - 1];
		for (size_t p = 0; p < numberLayer[l - 1].size(); ++p)
		{
			numberLayer[l][p]->input(prevLayer);
		}
	}
}

Neural::Neural(const vector<float> &diagram)
{
	size_t numlayers = diagram.size(); // numberlayers is the number of layers

	for (size_t i = 0; i < numlayers; i++)
	{
		numberLayer.push_back(Layer());

		int numberNeuronnextlayer = i == numlayers - 1 ? -1 : numlayers + 1; // if layer number is -1 for it will have one neuron , then exit else four neurons in each layer

/*
 * kbw: you need to rewrite this--I can't make sense of it.
 *		for (size_t j = 0; j <= numlayers; j++) // j is the number of neurons in each layer and one is bias
 *		{
 *			Layer().push_back(Neuron(numberNeuronnextlayer,j));
 *		}
 *		numlayers.back().back().setOutputVal(1.0);
 */
	}

	for (int i = 0; i < numlayers; i++)
	{
		for (int j = 0; j <= numlayers; j++)
		{
			//			cout << "life =" <<i<<""<<j<< endl;
		}
	}
}

void showVectorVals(string label, vector <float> &v)
{
	cout << label << " ";
	for (size_t i = 0; i < v.size(); ++i)
	{
		cout << v[i] << " ";
	}
	cout << endl;
}

//---------------------------------------------------------------------------
//
int main()
{
	TrainingData trainData("C:\\Users\\chica\\Downloads\\3rd sem\\OOPS\\final project\\trainingData.txt");

	vector<float> diagram;
	trainData.getDiagram(diagram);
	Neural myNeural(diagram);

	vector<float> inputVal, middleVal1, outputVal;
	int trainingPass = 0;

	while (!trainData.isEof())
	{
		++trainingPass;
		cout << endl << "Pass" << trainingPass;

		// Get new input data and forward feed
		if (trainData.getNextInputs(inputVal) != diagram[0])
		{
			break;
		}

		showVectorVals(": Inputs:", inputVal);
		myNeural.input(inputVal);

		// collect the net actual results:
		myNeural.output(outputVal);
		showVectorVals("Outputs:", outputVal);

		// Train the net what the ouputs should have been
		trainData.getTargetOutputs(middleVal1);
		showVectorVals("Targets:", middleVal1);
		assert(middleVal1.size() == diagram.size());

		myNeural.middle1(middleVal1);

		//Report how well the training is working averged
		cout << " net recent average error : " << myNeural.getrecentAverageError() << endl;
	}

	cout << endl << "Done" << endl;

	diagram.clear();
	//	vector<float> diagram;
		//diagram.push_back(4);
	diagram.push_back(3);
	diagram.push_back(2);
	diagram.push_back(1);
	Neural myNeural2(diagram);

	//	vector<double > inputVal;
	myNeural2.input(inputVal);

	//	vector<double > middleVal1;
	myNeural2.middle1(middleVal1);

	//vector<double > middleVal2;
	//myNeural2.middle2(middleVal2);

//	vector<double > outputVal;
	myNeural2.output(outputVal);
}
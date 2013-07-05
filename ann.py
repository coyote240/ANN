#!/usr/bin/python
#
#   ANN - Artificial Neural Network
#   $Id: ann.py 2 2005-05-05 03:48:18Z  $

import math, random

class Neuron:
    def __init__(self, inputCount):

        #   Init Obj Attrs
        self.inputs     = inputCount
        self.weights    = []
        self.threshold  = random.random()
        self.error      = 0
        self.actv       = 0

        #   Prepop neuron w/ random weights
        for i in range(self.inputs):
            self.weights.append(random.random())


class Network:
    def __init__(self, inputCount, outputCount, learningRate = 0.02):

        #   Seed Random
        random.seed()

        #   Init Obj Attrs
        self.inputCount     = inputCount
        self.outputCount    = outputCount
        self.learningRate   = learningRate

        #   Init network neurons
        self.inputLayer     = []
        self.hiddenLayer    = []
        self.outputLayer    = []

        for i in range(self.inputCount):
            self.inputLayer.append( Neuron(self.inputCount) )
            self.hiddenLayer.append( Neuron(self.inputCount) )

        for i in range(self.outputCount):
            self.outputLayer.append( Neuron(self.inputCount) )


    def run(self, testPattern, desiredOut = None):

        output = []

        #   Execute feed forward algorithm
        self.feedForward(testPattern)

        #   Train if desired output is present
        if desiredOut is not None:
            self.backProp(testPattern, desiredOut)

        #   Prepare returned array
        for neuron in self.outputLayer:
            output.append(neuron.a)

        return output


    def feedForward(self, testPattern):

        #   Process Input Layer
        for neuron in self.inputLayer:
            sum = 0
            for i in range(len(testPattern)):
                sum += neuron.weights[i] * testPattern[i]

            neuron.a = sigmoid(sum - neuron.threshold)

        #   Process Hidden Layer
        for neuron in self.hiddenLayer:
            sum = 0
            for i in range(len(self.inputLayer)):
                sum += neuron.weights[i] * self.inputLayer[i].a

            neuron.a = sigmoid(sum - neuron.threshold)

        #   Process Output Layer
        for neuron in self.outputLayer:
            sum = 0
            for i in range(len(self.hiddenLayer)):
                sum += neuron.weights[i] * self.hiddenLayer[i].a

            neuron.a = sigmoid(sum - neuron.threshold)


    def backProp(self, testPattern, desiredOut):

        #   Calc layer errors
        self.__calcOutputLayerErrors(desiredOut)
        self.__calcHiddenLayerErrors()
        self.__calcInputLayerErrors()

        #   Adjust weights and thresholds
        self.__trainOutputLayer()
        self.__trainHiddenLayer()
        self.__trainInputLayer(testPattern)


    def __calcOutputLayerErrors(self, desiredOut):
        for i in range(len(self.outputLayer)):
            a = self.outputLayer[i].a

            self.outputLayer[i].error = (desiredOut[i] - a) * a * (1 - a)


    def __calcHiddenLayerErrors(self):
        for i in range(len(self.hiddenLayer)):
            sum = 0

            for neuron in self.outputLayer:
                sum += neuron.error * neuron.weights[i]

            self.hiddenLayer[i].error = sum


    def __calcInputLayerErrors(self):
        for i in range(len(self.inputLayer)):
            sum = 0

            for neuron in self.hiddenLayer:
                sum += neuron.error * neuron.weights[i]

            self.inputLayer[i].error = sum


    def __trainOutputLayer(self):
        for i in range(len(self.outputLayer)):
            for j in range(len(self.hiddenLayer)):

                #   Calc and assign new weight
                weight = self.outputLayer[i].weights[j]
                weight -= self.learningRate * self.hiddenLayer[j].a
                self.outputLayer[i].weights[j] = weight

            #   Calc and assign new threshold
            threshold = self.outputLayer[i].threshold
            threshold -= self.learningRate * self.outputLayer[i].error
            self.outputLayer[i].threshold = threshold


    def __trainHiddenLayer(self):
        for i in range(len(self.hiddenLayer)):
            for j in range(len(self.inputLayer)):

                #   Calc and assign new weight
                weight = self.hiddenLayer[i].weights[j]
                weight += self.learningRate * self.inputLayer[j].a
                self.hiddenLayer[i].weights[j] = weight

            #   Calc and assign new threshold
            threshold = self.hiddenLayer[i].threshold
            threshold -= self.learningRate * self.hiddenLayer[i].error
            self.hiddenLayer[i].threshold = threshold


    def __trainInputLayer(self, testPattern):
        for i in range(len(self.inputLayer)):
            for j in range(len(testPattern)):

                #   Calc and assign new weight
                weight = self.inputLayer[i].weights[j]
                weight += self.learningRate * testPattern[j]
                self.inputLayer[i].weights[j] = weight

            #   Calc and assign new threshold
            threshold = self.inputLayer[i].threshold
            threshold -= self.learningRate * self.inputLayer[i].error
            self.inputLayer[i].threshold = threshold


def sigmoid(x):
    return math.tanh(x)

package ai.greatfree.linearregression;

import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;

import java.io.IOException;

import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

/**
 * Created: 08/14/2024, Bing Li
 */
final class SoftmaxRegressionScratch
{
	public static void main(String[] args) throws IOException, TranslateException
	{
		int batchSize = 256;
		boolean randomShuffle = true;

		// get training and validation dataset
		FashionMnist trainingSet = FashionMnist.builder().optUsage(Dataset.Usage.TRAIN).setSampling(batchSize, randomShuffle).optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE)).build();

		FashionMnist validationSet = FashionMnist.builder().optUsage(Dataset.Usage.TEST).setSampling(batchSize, false).optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE)).build();

		try (NDManager manager = NDManager.newBaseManager())
		{
			// int numInputs = 784;
			// int numOutputs = 10;

			SoftMaxProfile.TRAIN().setManager(manager);

			NDArray w = manager.randomNormal(0, 0.01f, new Shape(SoftMaxProfile.TRAIN().getNumInputs(), SoftMaxProfile.TRAIN().getNumOutputs()), DataType.FLOAT32);
			NDArray b = manager.zeros(new Shape(SoftMaxProfile.TRAIN().getNumOutputs()), DataType.FLOAT32);
			NDList params = new NDList(w, b);
			SoftMaxProfile.TRAIN().setParams(params);

			// int numEpochs = 5;
			// float lr = 0.1f;
			// trainCh3(manager, Net::net, trainingSet, validationSet,
			// LossFunction::crossEntropy, numEpochs, Updater::updater, params,
			// SoftMaxProfile.TRAIN().getNumInputs(), lr);
			LNTraining.trainCh3(Net::net, trainingSet, validationSet, LossFunction::crossEntropy, SoftMaxProfile.TRAIN().getNumEpochs(), Updater::updater);
			LNTraining.predictCh3(Net::net, validationSet, 6, manager);
		}
	}
}

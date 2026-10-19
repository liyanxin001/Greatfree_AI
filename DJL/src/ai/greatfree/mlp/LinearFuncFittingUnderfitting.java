package ai.greatfree.mlp;

import java.io.IOException;

import org.apache.commons.math3.special.Gamma;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.TranslateException;

/**
 * Created: 08/17/2024, Bing Li
 */
final class LinearFuncFittingUnderfitting
{

	public static void main(String[] args) throws IOException, TranslateException
	{
		int maxDegree = 20; // Maximum degree of the polynomial
		// Training and test dataset sizes
		int nTrain = 100;
		int nTest = 100;

		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray trueW = manager.zeros(new Shape(maxDegree)); // Allocate lots of empty space
			System.out.println("trueW = " + trueW);

			NDArray tempArr = manager.create(new float[] { 5f, 1.2f, -3.4f, 5.6f });
			System.out.println("--------------------");
			System.out.println("tempArr = " + tempArr);

			for (int i = 0; i < tempArr.size(); i++)
			{
				trueW.set(new NDIndex(i), tempArr.getFloat(i));
			}
			System.out.println("--------------------");
			System.out.println("trueW = " + trueW);

			NDArray features = manager.randomNormal(new Shape(nTrain + nTest, 1));
			System.out.println("--------------------");
			System.out.println("features = " + features);

			features = MLPUtils.shuffle(features);
			System.out.println("--------------------");
			System.out.println("features = " + features);

			NDArray polyFeatures = features.pow(manager.arange(maxDegree).reshape(1, -1));
			System.out.println("--------------------");
			System.out.println("polyFeatures = " + polyFeatures);

			for (int i = 0; i < maxDegree; i++)
			{
				polyFeatures.set(new NDIndex(":, " + i), polyFeatures.get(":, " + i).div(Gamma.gamma(i + 1)));
			}
			System.out.println("--------------------");
			System.out.println("polyFeatures = " + polyFeatures);

			// NDArray factorialArr =
			// factorial(manager.arange(maxDegree).add(1.0f).toType(DataType.FLOAT32,
			// false)).reshape(1, -1);

			// polyFeatures = polyFeatures.div(factorialArr);
			// Shape of `labels`: (`n_train` + `n_test`,)
			NDArray labels = polyFeatures.dot(trueW);
			System.out.println("--------------------");
			System.out.println("labels = " + labels);

			labels = labels.add(manager.randomNormal(0, 0.1f, labels.getShape(), DataType.FLOAT32));
			System.out.println("--------------------");
			System.out.println("labels = " + labels);

			System.out.println("--------------------");
			System.out.println("features: " + features.get(":2"));
			System.out.println("polyFeatures: " + polyFeatures.get(":2"));
			System.out.println("labels: " + labels.get(":2"));

			int logInterval = 20;
			int numEpochs = Integer.getInteger("MAX_EPOCH", 400);

			double[] trainLoss = new double[numEpochs/logInterval];
			double[] testLoss = new double[numEpochs/logInterval];
			double[] epochCount = new double[numEpochs/logInterval];
			
			int nDegree = 2;
			MLPUtils.train("Underfitting", polyFeatures.get("0:" + nTrain + ", 0:" + nDegree), polyFeatures.get(nTrain + ": , 0:" + nDegree), labels.get(":" + nTrain), labels.get(nTrain + ":"), nDegree, logInterval, numEpochs, trainLoss, testLoss, epochCount);
		}
	}
}


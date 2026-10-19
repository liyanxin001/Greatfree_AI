package ai.greatfree.convolutionnn;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.lang3.ArrayUtils;

import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.translate.TranslateException;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.LinePlot;

/**
 * Created: 08/20/2024, Bing Li
 */
class ConvolutionUtils
{
	public static NDArray corr2d(NDManager manager, NDArray X, NDArray K)
	{
		// Compute 2D cross-correlation.
		int h = (int) K.getShape().get(0);
		int w = (int) K.getShape().get(1);

		NDArray Y = manager.zeros(new Shape(X.getShape().get(0) - h + 1, X.getShape().get(1) - w + 1));

		for (int i = 0; i < Y.getShape().get(0); i++)
		{
			for (int j = 0; j < Y.getShape().get(1); j++)
			{
				Y.set(new NDIndex(i + "," + j), X.get(i + ":" + (i + h) + "," + j + ":" + (j + w)).mul(K).sum());
			}
		}
		return Y;
	}

	public static NDArray corr2dMultiIn(NDManager manager, NDArray X, NDArray K)
	{
		long h = K.getShape().get(0);
		long w = K.getShape().get(1);

		// First, traverse along the 0th dimension (channel dimension) of `X` and
		// `K`. Then, add them together

		NDArray res = manager.zeros(new Shape(X.getShape().get(0) - h + 1, X.getShape().get(1) - w + 1));
		for (int i = 0; i < X.getShape().get(0); i++)
		{
			for (int j = 0; j < K.getShape().get(0); j++)
			{
				if (i == j)
				{
					res = res.add(corr2d(manager, X.get(new NDIndex(i)), K.get(new NDIndex(j))));
				}
			}
		}
		return res;
	}

	public static NDArray corrMultiInOut(NDManager manager, NDArray X, NDArray K)
	{
		long cin = K.getShape().get(0);
		long h = K.getShape().get(2);
		long w = K.getShape().get(3);

		// Traverse along the 0th dimension of `K`, and each time, perform
		// cross-correlation operations with input `X`. All of the results are
		// merged together using the stack function
		NDArray res = manager.create(new Shape(cin, X.getShape().get(1) - h + 1, X.getShape().get(2) - w + 1));
		for (int j = 0; j < K.getShape().get(0); j++)
		{
			res.set(new NDIndex(j), corr2dMultiIn(manager, X, K.get(new NDIndex(j))));
		}
		return res;
	}

	public static NDArray corr2dMultiInOut1x1(NDArray X, NDArray K)
	{
		long channelIn = X.getShape().get(0);
		long height = X.getShape().get(1);
		long width = X.getShape().get(2);

		long channelOut = K.getShape().get(0);
		X = X.reshape(channelIn, height * width);
		K = K.reshape(channelOut, channelIn);
		NDArray Y = K.dot(X); // Matrix multiplication in the fully connected layer
		return Y.reshape(channelOut, height, width);
	}

	public static NDArray pool2d(NDManager manager, NDArray X, Shape poolShape, String mode)
	{
		long poolHeight = poolShape.get(0);
		long poolWidth = poolShape.get(1);

		NDArray Y = manager.zeros(new Shape(X.getShape().get(0) - poolHeight + 1, X.getShape().get(1) - poolWidth + 1));
		for (int i = 0; i < Y.getShape().get(0); i++)
		{
			for (int j = 0; j < Y.getShape().get(1); j++)
			{
				if ("max".equals(mode))
				{
					Y.set(new NDIndex(i + "," + j), X.get(new NDIndex(i + ":" + (i + poolHeight) + ", " + j + ":" + (j + poolWidth))).max());
				}
				else if ("avg".equals(mode))
				{
					Y.set(new NDIndex(i + "," + j), X.get(new NDIndex(i + ":" + (i + poolHeight) + ", " + j + ":" + (j + poolWidth))).mean());
				}
			}
		}
		return Y;
	}

	public static void trainViaLeNet(ArrayDataset trainIter, ArrayDataset testIter, int numEpochs, Trainer trainer, double[] epochCount) throws IOException, TranslateException
	{
		double avgTrainTimePerEpoch = 0;
		Map<String, double[]> evaluatorMetrics = new HashMap<>();
		trainer.setMetrics(new Metrics());

		EasyTrain.fit(trainer, numEpochs, trainIter, testIter);

		Metrics metrics = trainer.getMetrics();

		trainer.getEvaluators().stream().forEach(evaluator ->
		{
			evaluatorMetrics.put("train_epoch_" + evaluator.getName(), metrics.getMetric("train_epoch_" + evaluator.getName()).stream().mapToDouble(x -> x.getValue().doubleValue()).toArray());
			evaluatorMetrics.put("validate_epoch_" + evaluator.getName(), metrics.getMetric("validate_epoch_" + evaluator.getName()).stream().mapToDouble(x -> x.getValue().doubleValue()).toArray());
		});

		avgTrainTimePerEpoch = metrics.mean("epoch");

		double[] trainLoss = evaluatorMetrics.get("train_epoch_SoftmaxCrossEntropyLoss");
		double[] trainAccuracy = evaluatorMetrics.get("train_epoch_Accuracy");
		double[] testAccuracy = evaluatorMetrics.get("validate_epoch_Accuracy");

		System.out.printf("loss %.3f,", trainLoss[numEpochs - 1]);
		System.out.printf(" train acc %.3f,", trainAccuracy[numEpochs - 1]);
		System.out.printf(" test acc %.3f\n", testAccuracy[numEpochs - 1]);
		System.out.printf("%.1f examples/sec \n", trainIter.size() / (avgTrainTimePerEpoch / Math.pow(10, 9)));

		String[] lossLabel = new String[trainLoss.length + testAccuracy.length + trainAccuracy.length];

		Arrays.fill(lossLabel, 0, trainLoss.length, "train loss");
		Arrays.fill(lossLabel, trainAccuracy.length, trainLoss.length + trainAccuracy.length, "train acc");
		Arrays.fill(lossLabel, trainLoss.length + trainAccuracy.length, trainLoss.length + testAccuracy.length + trainAccuracy.length, "test acc");

		Table data = Table.create("Data").addColumns(DoubleColumn.create("epoch", ArrayUtils.addAll(epochCount, ArrayUtils.addAll(epochCount, epochCount))), DoubleColumn.create("metrics", ArrayUtils.addAll(trainLoss, ArrayUtils.addAll(trainAccuracy, testAccuracy))), StringColumn.create("lossLabel", lossLabel));

//		render(LinePlot.create("", data, "epoch", "metrics", "lossLabel"), "text/html");
		Plot.show(LinePlot.create("", data, "epoch", "metrics", "lossLabel"));
	}
}

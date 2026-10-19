package ai.greatfree.performance;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import ai.djl.Device;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.translate.TranslateException;

/**
 * Created: 08/22/2024, Bing Li
 */
final class PerUtils
{
	public static NDArray run(NDArray X)
	{
		for (int i = 0; i < 10; i++)
		{
			X = X.dot(X);
		}
		return X;
	}

	public static NDArray copyToCPU(NDArray X)
	{
		NDArray Y = X.toDevice(Device.cpu(), true);
		return Y;
	}

	public static SequentialBlock resnetBlock(int numChannels, int numResiduals, boolean isFirstBlock)
	{
		SequentialBlock blk = new SequentialBlock();
		for (int i = 0; i < numResiduals; i++)
		{

			if (i == 0 && !isFirstBlock)
			{
				blk.add(new Residual(numChannels, true, new Shape(2, 2)));
			}
			else
			{
				blk.add(new Residual(numChannels, false, new Shape(1, 1)));
			}
		}
		return blk;
	}

	public static double[] train(int numEpochs, Trainer trainer, int batchSize) throws IOException, TranslateException
	{
		FashionMnist trainIter = FashionMnist.builder().optUsage(Dataset.Usage.TRAIN).setSampling(batchSize, true).optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE)).build();
		FashionMnist testIter = FashionMnist.builder().optUsage(Dataset.Usage.TEST).setSampling(batchSize, true).optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE)).build();

		trainIter.prepare();
		testIter.prepare();
		
		Map<String, double[]> evaluatorMetrics = new HashMap<>();
		double avgTrainTime = 0;

		trainer.setMetrics(new Metrics());

		EasyTrain.fit(trainer, numEpochs, trainIter, testIter);

		Metrics metrics = trainer.getMetrics();

		trainer.getEvaluators().stream().forEach(evaluator ->
		{
			evaluatorMetrics.put("validate_epoch_" + evaluator.getName(), metrics.getMetric("validate_epoch_" + evaluator.getName()).stream().mapToDouble(x -> x.getValue().doubleValue()).toArray());
		});

		avgTrainTime = metrics.mean("epoch");
		double[] testAccuracy = evaluatorMetrics.get("validate_epoch_Accuracy");
		System.out.printf("test acc %.2f\n", testAccuracy[numEpochs - 1]);
		System.out.println(avgTrainTime / Math.pow(10, 9) + " sec/epoch \n");
		return testAccuracy;
	}
}
